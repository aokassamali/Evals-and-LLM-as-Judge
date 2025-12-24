import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def safe_name_from_runpath(run_path: str) -> str:
    # e.g. runs/scifact_dev200_qwen3_8b_t0.0.jsonl -> qwen3_8b_t0.0
    base = os.path.basename(run_path)
    base = base.replace(".jsonl", "")
    for prefix in ["scifact_dev200_", "run_", "scifact_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="Gold JSONL (SciFact oracle) path")
    p.add_argument("--runs_dir", default="runs", help="Directory containing run JSONLs")
    p.add_argument("--pattern", default="scifact_dev200_*.jsonl", help="Glob pattern inside runs_dir")
    p.add_argument("--metrics_script", default="scripts/compute_single_model_metrics.py", help="Path to compute_single_model_metrics.py")

    # Outputs (default to results/)
    p.add_argument("--out_dir", default="results/metrics_json", help="Where per-run metrics JSONs go")
    p.add_argument("--summary_json", default="results/metrics_table.json", help="Combined summary JSON")
    p.add_argument("--summary_md", default="results/metrics_table.md", help="Summary markdown table")
    p.add_argument("--summary_csv", default="results/metrics_table.csv", help="Summary CSV table")
    p.add_argument("--label_breakdown_json", default="results/label_breakdown.json", help="Per-class label metrics JSON")
    args = p.parse_args()

    runs_glob = str(Path(args.runs_dir) / args.pattern)
    run_paths = sorted(glob.glob(runs_glob))
    if not run_paths:
        print(f"No run files matched: {runs_glob}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    label_breakdown = {}  # model -> per_class dict

    for run_path in run_paths:
        name = safe_name_from_runpath(run_path)
        out_path = out_dir / f"{name}.json"

        cmd = [
            sys.executable,
            args.metrics_script,
            "--gold", args.gold,
            "--run", run_path,
            "--out", str(out_path),
        ]
        print("\n== Metrics for:", run_path)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! metrics failed for {run_path}")
            return r.returncode

        m = json.loads(out_path.read_text(encoding="utf-8"))
        lab = m.get("label", {}) or {}
        ev = m.get("evidence", {}) or {}

        # pull optional diagnostics if present
        summary_rows.append({
            "model": name,
            "run_path": run_path,
            "metrics_path": str(out_path),
            "n_errors_in_run": m.get("n_errors_in_run", None),

            "label_n": lab.get("n_scored", None),
            "label_accuracy": lab.get("accuracy", None),
            "label_macro_f1": lab.get("macro_f1", None),

            "evidence_n": ev.get("n_scored", None),
            "evidence_precision": ev.get("mean_precision", None),
            "evidence_recall": ev.get("mean_recall", None),
            "evidence_f1": ev.get("mean_f1", None),

            # Phase 3 diagnostics
            "avg_predicted_sentences": ev.get("avg_predicted_sentences", None),
            "pct_empty_when_pred_non_nei": ev.get("pct_empty_when_pred_non_nei", None),
            "pct_nonempty_when_pred_nei": ev.get("pct_nonempty_when_pred_nei", None),
            "pct_cite_all_unnecessary": ev.get("pct_cite_all_unnecessary", None),
            "pct_pred_shotgun": ev.get("pct_pred_shotgun", None),

            "correct_label_wrong_evidence": ev.get("correct_label_wrong_evidence", None),
        })

        if isinstance(lab.get("per_class"), dict):
            label_breakdown[name] = lab["per_class"]

    # --- write JSON summary ---
    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps({"rows": summary_rows}, indent=2), encoding="utf-8")
    print("\nWrote:", summary_json_path)

    # --- write label breakdown ---
    lb_path = Path(args.label_breakdown_json)
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    lb_path.write_text(json.dumps(label_breakdown, indent=2), encoding="utf-8")
    print("Wrote:", lb_path)

    # --- write markdown table ---
    md_lines = []
    md_lines.append("|model|acc|macro_f1|ev_P|ev_R|ev_F1|avg_ev|%empty_when_not_NEI|%NOT_empty_when_NEI|%cite_all_unnecessary|%shotgun|correct_label_wrong_ev|errors|n|")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in summary_rows:
        md_lines.append(
            f"|{r['model']}|{fmt(r['label_accuracy'])}|{fmt(r['label_macro_f1'])}"
            f"|{fmt(r['evidence_precision'])}|{fmt(r['evidence_recall'])}|{fmt(r['evidence_f1'])}"
            f"|{fmt(r['avg_predicted_sentences'])}"
            f"|{fmt(r['pct_empty_when_pred_non_nei'])}|{fmt(r['pct_nonempty_when_pred_nei'])}"
            f"|{fmt(r['pct_cite_all_unnecessary'])}|{fmt(r['pct_pred_shotgun'])}"
            f"|{r['correct_label_wrong_evidence'] if r['correct_label_wrong_evidence'] is not None else ''}"
            f"|{r['n_errors_in_run']}|{r['label_n']}|"
        )

    summary_md_path = Path(args.summary_md)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Wrote:", summary_md_path)

    # --- write CSV table ---
    summary_csv_path = Path(args.summary_csv)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print("Wrote:", summary_csv_path)

    print("\n--- Summary table ---\n")
    print("\n".join(md_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
