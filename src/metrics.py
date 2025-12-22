import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

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


def get_pred_evidence_ids(run_row: Dict[str, Any]) -> Optional[Set[int]]:
    """
    Pull predicted evidence sentence IDs from run_row["validated"]["evidence_sent_ids"].
    Returns a set[int] or None if missing/invalid.
    """
    v = run_row.get("validated")
    if not isinstance(v, dict):
        return None
    ids = v.get("evidence_sent_ids")
    if not isinstance(ids, list):
        return None
    out: Set[int] = set()
    for x in ids:
        if isinstance(x, int):
            out.add(x)
        elif isinstance(x, str) and x.isdigit():
            out.add(int(x))
        else:
            # ignore weird entries rather than failing the whole row
            continue
    return out


def get_gold_evidence_ids(gold_row: Dict[str, Any]) -> Optional[Set[int]]:
    """
    Pull gold evidence sentence IDs from gold_row["gold_evidence_sent_ids"].
    Returns a set[int] or None if missing/invalid.
    """
    ids = gold_row.get("gold_evidence_sent_ids")
    if not isinstance(ids, list):
        return None
    out: Set[int] = set()
    for x in ids:
        if isinstance(x, int):
            out.add(x)
        elif isinstance(x, str) and x.isdigit():
            out.add(int(x))
        else:
            continue
    return out


def precision_recall_f1(pred: Set[int], gold: Set[int]) -> Tuple[float, float, float]:
    """
    Strict evidence scoring:
      precision = |pred ∩ gold| / |pred|   (if pred empty -> 1 if gold empty else 0)
      recall    = |pred ∩ gold| / |gold|   (if gold empty -> 1)
      f1        = harmonic mean (if both precision+recall==0 -> 0)
    """
    inter = len(pred & gold)

    if len(pred) == 0:
        precision = 1.0 if len(gold) == 0 else 0.0
    else:
        precision = inter / len(pred)

    if len(gold) == 0:
        recall = 1.0
    else:
        recall = inter / len(gold)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="Input JSONL with gold labels/evidence (id, gold_label, gold_evidence_sent_ids).")
    parser.add_argument("--run", type=str, required=True, help="Run JSONL produced by run_batch.py.")
    parser.add_argument("--out", type=str, default="", help="Optional path to write metrics JSON.")
    args = parser.parse_args()

    gold_by_id = index_gold_by_id(Path(args.gold))

    labels = ["SUPPORTS", "REFUTES", "NEI"]
    y_true = []
    y_pred = []

    counts = Counter()
    errors = 0

    # Evidence aggregates
    ev_prec_sum = 0.0
    ev_rec_sum = 0.0
    ev_f1_sum = 0.0
    ev_scored = 0

    for rr in read_jsonl(Path(args.run)):
        rid = rr.get("id")
        if not isinstance(rid, str) or rid not in gold_by_id:
            continue

        gold_row = gold_by_id[rid]
        gold_lab = normalize_gold_label(gold_row.get("gold_label"))
        pred_lab = get_pred_label(rr)

        if rr.get("error") is not None:
            errors += 1
            continue

        # --- Label scoring ---
        if gold_lab not in labels or pred_lab not in labels:
            counts["skipped_bad_label"] += 1
        else:
            y_true.append(gold_lab)
            y_pred.append(pred_lab)
            counts["label_scored"] += 1

        # --- Evidence scoring (independent of label scoring) ---
        gold_ev = get_gold_evidence_ids(gold_row)
        pred_ev = get_pred_evidence_ids(rr)

        if gold_ev is None:
            counts["evidence_missing_gold"] += 1
        elif pred_ev is None:
            counts["evidence_missing_pred"] += 1
        else:
            p, r, f1 = precision_recall_f1(pred_ev, gold_ev)
            ev_prec_sum += p
            ev_rec_sum += r
            ev_f1_sum += f1
            ev_scored += 1
            counts["evidence_scored"] += 1

    if len(y_true) == 0 and ev_scored == 0:
        raise RuntimeError("No scored examples. Check IDs match and gold fields are present.")

    report: Dict[str, Any] = {
        "n_errors_in_run": errors,
        "counts": dict(counts),
    }

    # Label metrics (only if we scored any)
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report.update(
            {
                "label": {
                    "n_scored": len(y_true),
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "labels_order": labels,
                    "confusion_matrix": cm,
                }
            }
        )

    # Evidence metrics (only if we scored any)
    if ev_scored > 0:
        report.update(
            {
                "evidence": {
                    "n_scored": ev_scored,
                    "mean_precision": ev_prec_sum / ev_scored,
                    "mean_recall": ev_rec_sum / ev_scored,
                    "mean_f1": ev_f1_sum / ev_scored,
                }
            }
        )

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
