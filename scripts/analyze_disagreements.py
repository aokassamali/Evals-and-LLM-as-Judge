import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no} in {path}: {e}")


def normalize_label(x: Any) -> str:
    if not isinstance(x, str):
        return "INVALID"
    s = x.strip().upper()
    mapping = {
        "SUPPORT": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "SUPPORTS": "SUPPORTS",
        "REFUTE": "REFUTES",
        "REFUTED": "REFUTES",
        "REFUTES": "REFUTES",
        "CONTRADICT": "REFUTES",
        "CONTRADICTION": "REFUTES",
        "NEI": "NEI",
        "NOT_ENOUGH_INFO": "NEI",
        "NOT ENOUGH INFO": "NEI",
        "INSUFFICIENT": "NEI",
    }
    return mapping.get(s, s)


def extract_validated(run_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    v = run_row.get("validated")
    return v if isinstance(v, dict) else None


def get_candidate_ids(gold_row: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    ev = gold_row.get("evidence")
    if isinstance(ev, list):
        for it in ev:
            if isinstance(it, dict) and isinstance(it.get("sent_id"), int):
                out.add(int(it["sent_id"]))
    return out


def get_gold_evidence_ids(gold_row: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    ids = gold_row.get("gold_evidence_sent_ids")
    if isinstance(ids, list):
        for x in ids:
            if isinstance(x, int):
                out.add(x)
            elif isinstance(x, str) and x.isdigit():
                out.add(int(x))
    return out


def get_pred_evidence_ids(validated: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    ids = validated.get("evidence_sent_ids")
    if isinstance(ids, list):
        for x in ids:
            if isinstance(x, int):
                out.add(x)
            elif isinstance(x, str) and x.isdigit():
                out.add(int(x))
    return out


def claim_flags(claim: str) -> Dict[str, bool]:
    c = claim.lower()
    numeric = bool(re.search(r"\b\d+(\.\d+)?\b", c)) or any(tok in c for tok in ["%", "percent", "mg", "ml", "kg"])
    causal = any(w in c for w in [
        "increase", "increases", "decrease", "decreases", "reduce", "reduces",
        "cause", "causes", "leads to", "results in", "improve", "improves",
        "worsen", "worsens", "affect", "affects"
    ])
    hedged = any(w in c for w in [
        "may", "might", "could", "suggest", "suggests", "likely", "possibly",
        "appears", "indicate", "indicates"
    ])
    return {"numeric": numeric, "causal": causal, "hedged": hedged}


def label_flip_bucket(a: str, b: str) -> str:
    if a == b:
        return "same"
    pair = {a, b}
    if pair == {"SUPPORTS", "REFUTES"}:
        return "SUPPORTS<->REFUTES"
    if "NEI" in pair and ("SUPPORTS" in pair or "REFUTES" in pair):
        return "NEI<->(S/R)"
    return "other"


def evidence_bucket(pred_ev: Set[int], cand_ids: Set[int]) -> str:
    n_pred = len(pred_ev)
    n_cand = len(cand_ids)
    if n_pred == 0:
        return "empty"
    if n_cand > 0 and n_pred == n_cand:
        return "cite_all"
    if n_cand > 0 and n_pred >= max(5, int(0.5 * n_cand)):
        return "shotgun"
    return "normal"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="Gold JSONL path")
    p.add_argument("--runs", nargs="*", default=None, help="Run JSONL files (optional if using --runs_dir)")
    p.add_argument("--runs_dir", default=None, help="Directory with run JSONLs")
    p.add_argument("--glob", default="scifact_dev200_*_t0.0.jsonl", help="Glob inside --runs_dir")
    p.add_argument("--names", nargs="*", default=None, help="Optional names for runs (same order)")
    p.add_argument("--ref", default=None, help="Reference model name (default: first)")
    p.add_argument("--out_md", default="results/disagreement_summary.md")
    p.add_argument("--out_sample", default="results/disagreements_sample.jsonl")
    p.add_argument("--k_examples", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    # Resolve run paths
    run_paths: List[Path] = []
    if args.runs_dir:
        run_paths = sorted([p_ for p_ in Path(args.runs_dir).glob(args.glob) if p_.is_file()])
    else:
        run_paths = [Path(x) for x in (args.runs or [])]

    if not run_paths:
        raise SystemExit("No run files found. Provide --runs <files...> or --runs_dir <dir> with --glob.")

    run_names = args.names if args.names else [p_.stem for p_ in run_paths]
    if len(run_names) != len(run_paths):
        raise SystemExit("If provided, --names must match number of run files.")

    rng = random.Random(args.seed)

    # Load gold by id
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for g in read_jsonl(Path(args.gold)):
        rid = g.get("id")
        if isinstance(rid, str):
            gold_by_id[rid] = g

    # Load each run: id -> {label, ev_set}
    run_data = {}
    for name, rp in zip(run_names, run_paths):
        by_id = {}
        n_err = 0
        for row in read_jsonl(rp):
            rid = row.get("id")
            if not isinstance(rid, str):
                continue
            if row.get("error"):
                n_err += 1
                continue
            v = extract_validated(row)
            if not v:
                n_err += 1
                continue
            lab = normalize_label(v.get("label"))
            ev = get_pred_evidence_ids(v)
            by_id[rid] = {"label": lab, "ev": ev}
        run_data[name] = {"by_id": by_id, "n_err": n_err}

    ref_name = args.ref or run_names[0]
    if ref_name not in run_data:
        raise SystemExit(f"ref={ref_name} not found. Available: {run_names}")

    # Shared IDs across gold + all runs
    shared = [rid for rid in gold_by_id.keys() if all(rid in run_data[n]["by_id"] for n in run_names)]
    shared.sort()

    out_counts = {}
    samples: List[Dict[str, Any]] = []

    for other in run_names:
        if other == ref_name:
            continue

        c_label = Counter()
        c_ev = Counter()
        c_flags = Counter()
        c_wrong_ev_correct_label = 0
        examples = defaultdict(list)

        for rid in shared:
            g = gold_by_id[rid]
            claim = g.get("claim") if isinstance(g.get("claim"), str) else ""
            flags = claim_flags(claim)

            cand_ids = get_candidate_ids(g)
            gold_lab = normalize_label(g.get("gold_label"))
            gold_ev = get_gold_evidence_ids(g)

            ref_lab = run_data[ref_name]["by_id"][rid]["label"]
            other_lab = run_data[other]["by_id"][rid]["label"]

            other_ev = run_data[other]["by_id"][rid]["ev"]

            flip = label_flip_bucket(ref_lab, other_lab)
            c_label[flip] += 1

            eb = evidence_bucket(other_ev, cand_ids)
            c_ev[eb] += 1

            if other_lab == gold_lab and len(gold_ev) > 0 and len(other_ev & gold_ev) == 0:
                c_wrong_ev_correct_label += 1

            if ref_lab != other_lab:
                for k, v in flags.items():
                    if v:
                        c_flags[k] += 1
                examples[flip].append({
                    "id": rid,
                    "gold_label": gold_lab,
                    "ref_label": ref_lab,
                    "other_label": other_lab,
                    "other_ev_bucket": eb,
                    "other_n_ev": len(other_ev),
                    "claim": claim,
                })
                samples.append({
                    "id": rid,
                    "gold_label": gold_lab,
                    ref_name: ref_lab,
                    other: other_lab,
                })

        picked = {}
        for bucket in ["SUPPORTS<->REFUTES", "NEI<->(S/R)", "other"]:
            exs = examples.get(bucket, [])
            rng.shuffle(exs)
            picked[bucket] = exs[: args.k_examples]

        out_counts[other] = {
            "n": len(shared),
            "label_flip": dict(c_label),
            "evidence_behavior": dict(c_ev),
            "claim_flags_among_disagreements": dict(c_flags),
            "correct_label_wrong_evidence": c_wrong_ev_correct_label,
            "examples": picked,
        }

    out_sample = Path(args.out_sample)
    out_sample.parent.mkdir(parents=True, exist_ok=True)
    with out_sample.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Disagreement summary (ref = {ref_name})")
    lines.append("")
    lines.append(f"Shared IDs across gold + {len(run_names)} runs: **{len(shared)}**")
    lines.append("")

    for other, d in out_counts.items():
        lines.append(f"## {other} vs {ref_name}")
        lines.append("")
        lines.append(f"- n: **{d['n']}**")
        lines.append(f"- label flips: `{d['label_flip']}`")
        lines.append(f"- evidence behavior: `{d['evidence_behavior']}`")
        lines.append(f"- claim flags among disagreements: `{d['claim_flags_among_disagreements']}`")
        lines.append(f"- correct label but wrong evidence (vs gold): **{d['correct_label_wrong_evidence']}**")
        lines.append("")
        for bucket, exs in d["examples"].items():
            if not exs:
                continue
            lines.append(f"### Examples: {bucket}")
            lines.append("")
            for ex in exs:
                claim = (ex["claim"] or "").strip()
                if len(claim) > 220:
                    claim = claim[:220] + "…"
                lines.append(f"- `{ex['id']}` gold={ex['gold_label']}  ref={ex['ref_label']}  {other}={ex['other_label']}  other_ev={ex['other_n_ev']} ({ex['other_ev_bucket']})")
                lines.append(f"  - {claim}")
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote:", out_md)
    print("Wrote:", out_sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
