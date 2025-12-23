import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def safe_name_from_runpath(run_path: str) -> str:
    # e.g. runs/scifact_dev200_qwen3_8b_t0.jsonl -> qwen3_8b_t0
    base = os.path.basename(run_path)
    base = base.replace(".jsonl", "")
    # strip common prefix
    for prefix in ["scifact_dev200_", "run_", "scifact_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return base


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="Gold JSONL (SciFact oracle) path")
    p.add_argument("--runs_dir", default="runs", help="Directory containing run JSONLs")
    p.add_argument("--pattern", default="scifact_dev200_*.jsonl", help="Glob pattern inside runs_dir")
    p.add_argument("--metrics_script", default="src/metrics.py", help="Path to metrics.py")
    p.add_argument("--out_dir", default="metrics", help="Where to write metrics JSON files")
    p.add_argument("--summary_json", default="metrics/summary.json", help="Where to write combined summary JSON")
    p.add_argument("--summary_md", default="metrics/summary.md", help="Where to write markdown table")
    args = p.parse_args()

    runs_glob = str(Path(args.runs_dir) / args.pattern)
    run_paths = sorted(glob.glob(runs_glob))
    if not run_paths:
        print(f"No run files matched: {runs_glob}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for run_path in run_paths:
        name = safe_name_from_runpath(run_path)
        out_path = out_dir / f"{name}.json"

        cmd = [
            sys.executable, args.metrics_script,
            "--gold", args.gold,
            "--run", run_path,
            "--out", str(out_path),
        ]
        print("\n== Metrics for:", run_path)
        print("Cmd:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! metrics failed for {run_path}")
            return r.returncode

        m = json.loads(out_path.read_text(encoding="utf-8"))
        lab = m.get("label", {})
        ev = m.get("evidence", {})

        summary_rows.append({
            "name": name,
            "run_path": run_path,
            "metrics_path": str(out_path),
            "n_errors_in_run": m.get("n_errors_in_run", None),
            "label_n_scored": lab.get("n_scored", None),
            "label_accuracy": lab.get("accuracy", None),
            "label_macro_f1": lab.get("macro_f1", None),
            "evidence_n_scored": ev.get("n_scored", None),
            "evidence_precision": ev.get("mean_precision", None),
            "evidence_recall": ev.get("mean_recall", None),
            "evidence_f1": ev.get("mean_f1", None),
        })

    # Write combined JSON summary
    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps({"rows": summary_rows}, indent=2), encoding="utf-8")
    print("\nWrote:", summary_json_path)

    # Write markdown table
    md_lines = []
    md_lines.append("|model|acc|macro_f1|ev_P|ev_R|ev_F1|errors|n|")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        acc = r["label_accuracy"]
        mf1 = r["label_macro_f1"]
        ep = r["evidence_precision"]
        er = r["evidence_recall"]
        ef = r["evidence_f1"]
        errs = r["n_errors_in_run"]
        n = r["label_n_scored"]
        md_lines.append(
            f"|{r['name']}|{acc:.3f}|{mf1:.3f}|{ep:.3f}|{er:.3f}|{ef:.3f}|{errs}|{n}|"
        )

    summary_md_path = Path(args.summary_md)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Wrote:", summary_md_path)

    print("\n--- Summary table ---\n")
    print("\n".join(md_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
