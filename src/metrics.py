import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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


def index_gold_by_id(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    gold = {}
    for row in read_jsonl(gold_path):
        rid = row.get("id")
        if isinstance(rid, str):
            gold[rid] = row
    return gold


def get_pred_label(run_row: Dict[str, Any]) -> Optional[str]:
    v = run_row.get("validated")
    if isinstance(v, dict):
        lab = v.get("label")
        if isinstance(lab, str) and lab.strip():
            return lab.strip().upper()
    return None


def normalize_gold_label(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip().upper()
    mapping = {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NEI": "NEI",
        "NOT_ENOUGH_INFO": "NEI",
        "NOT ENOUGH INFO": "NEI",
    }
    return mapping.get(s, s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="Input JSONL with gold labels (id, gold_label, ...).")
    parser.add_argument("--run", type=str, required=True, help="Run JSONL produced by run_batch.py.")
    parser.add_argument("--out", type=str, default="", help="Optional path to write metrics JSON.")
    args = parser.parse_args()

    gold_by_id = index_gold_by_id(Path(args.gold))

    labels = ["SUPPORTS", "REFUTES", "NEI"]
    y_true = []
    y_pred = []

    counts = Counter()
    errors = 0

    for rr in read_jsonl(Path(args.run)):
        rid = rr.get("id")
        if not isinstance(rid, str) or rid not in gold_by_id:
            continue

        gold_lab = normalize_gold_label(gold_by_id[rid].get("gold_label"))
        pred_lab = get_pred_label(rr)

        if rr.get("error") is not None:
            errors += 1
            continue

        if gold_lab not in labels or pred_lab not in labels:
            counts["skipped_bad_label"] += 1
            continue

        y_true.append(gold_lab)
        y_pred.append(pred_lab)
        counts["scored"] += 1

    if not y_true:
        raise RuntimeError("No scored examples. Check that IDs match and gold_label is present.")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report = {
        "n_scored": len(y_true),
        "n_errors_in_run": errors,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels_order": labels,
        "confusion_matrix": cm,
        "counts": dict(counts),
    }

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
