import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


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
    gold: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(gold_path):
        rid = row.get("id")
        if isinstance(rid, str):
            gold[rid] = row
    return gold


def normalize_label(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip().upper()
    if not s:
        return None
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
        "INSUFFICIENT_EVIDENCE": "NEI",
    }
    return mapping.get(s, s)


def get_gold_label(gold_row: Dict[str, Any]) -> Optional[str]:
    return normalize_label(gold_row.get("gold_label"))


def get_pred_label(run_row: Dict[str, Any]) -> Optional[str]:
    v = run_row.get("validated")
    if not isinstance(v, dict):
        return None
    return normalize_label(v.get("label"))


def get_gold_evidence_ids(gold_row: Dict[str, Any]) -> Optional[Set[int]]:
    ids = gold_row.get("gold_evidence_sent_ids")
    if not isinstance(ids, list):
        return None
    out: Set[int] = set()
    for x in ids:
        if isinstance(x, int):
            out.add(x)
        elif isinstance(x, str) and x.isdigit():
            out.add(int(x))
    return out


def get_pred_evidence_ids(run_row: Dict[str, Any]) -> Optional[Set[int]]:
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
    return out


def get_candidate_evidence_ids(gold_row: Dict[str, Any]) -> Optional[Set[int]]:
    ev = gold_row.get("evidence")
    if not isinstance(ev, list):
        return None
    out: Set[int] = set()
    for item in ev:
        if isinstance(item, dict) and isinstance(item.get("sent_id"), int):
            out.add(int(item["sent_id"]))
    return out


def precision_recall_f1(pred: Set[int], gold: Set[int]) -> Tuple[float, float, float]:
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
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    gold_by_id = index_gold_by_id(Path(args.gold))
    labels = ["SUPPORTS", "REFUTES", "NEI"]

    # Label aggregates
    y_true = []
    y_pred = []
    pred_counts = Counter()
    true_counts = Counter()

    # Evidence aggregates
    ev_scored = 0
    ev_prec_sum = 0.0
    ev_rec_sum = 0.0
    ev_f1_sum = 0.0

    total_pred_ev = 0
    total_gold_ev = 0
    total_cand_ev = 0

    pred_empty_overall = 0
    cite_all_unnecessary = 0
    shotgun = 0

    # NEW: conditional “sanity check” counters based on predicted label
    n_pred_nei = 0
    n_pred_non_nei = 0
    nonempty_when_pred_nei = 0
    empty_when_pred_non_nei = 0

    correct_label_wrong_evidence = 0
    wrong_label_right_evidence = 0

    n_errors_in_run = 0
    n_missing_gold = 0

    # Tunable thresholds for "shotgun"
    PREC_CUTOFF = 0.7
    EXTRA_ABS = 2
    EXTRA_MULT = 1.5

    for row in read_jsonl(Path(args.run)):
        rid = row.get("id")
        if not isinstance(rid, str):
            n_errors_in_run += 1
            continue
        if row.get("error"):
            n_errors_in_run += 1
            continue

        gold_row = gold_by_id.get(rid)
        if gold_row is None:
            n_missing_gold += 1
            continue

        gold_lab = get_gold_label(gold_row)
        pred_lab = get_pred_label(row)

        # ---- label scoring ----
        if gold_lab is not None and pred_lab is not None:
            y_true.append(gold_lab)
            y_pred.append(pred_lab)
            pred_counts[pred_lab] += 1
            true_counts[gold_lab] += 1
        else:
            n_errors_in_run += 1

        # ---- evidence scoring ----
        gold_ev = get_gold_evidence_ids(gold_row)
        pred_ev = get_pred_evidence_ids(row)
        cand_ev = get_candidate_evidence_ids(gold_row)

        if gold_ev is None or pred_ev is None or cand_ev is None:
            continue

        p, r, f1 = precision_recall_f1(pred_ev, gold_ev)
        ev_prec_sum += p
        ev_rec_sum += r
        ev_f1_sum += f1
        ev_scored += 1

        K = len(gold_ev)
        P = len(pred_ev)
        C = len(cand_ev)

        total_pred_ev += P
        total_gold_ev += K
        total_cand_ev += C

        if P == 0:
            pred_empty_overall += 1

        # NEW: conditional sanity checks (use predicted label)
        if pred_lab == "NEI":
            n_pred_nei += 1
            if P > 0:
                nonempty_when_pred_nei += 1
        elif pred_lab in ("SUPPORTS", "REFUTES"):
            n_pred_non_nei += 1
            if P == 0:
                empty_when_pred_non_nei += 1

        # Cite-all only counts when unnecessary
        if C > 0 and P == C and K < C:
            cite_all_unnecessary += 1

        # Gold-aware shotgun: over-cite vs gold AND low precision
        overcite_threshold = max(K + EXTRA_ABS, int(math.ceil(EXTRA_MULT * max(1, K))))
        if P > overcite_threshold and p < PREC_CUTOFF:
            shotgun += 1

        # Cross-bucket diagnostics vs gold
        if gold_lab is not None and pred_lab is not None:
            label_correct = (gold_lab == pred_lab)
            evidence_good = (K == 0) or (len(pred_ev & gold_ev) > 0)

            if label_correct and (not evidence_good) and K > 0:
                correct_label_wrong_evidence += 1
            if (not label_correct) and evidence_good:
                wrong_label_right_evidence += 1

    report: Dict[str, Any] = {
        "n_errors_in_run": int(n_errors_in_run),
        "n_missing_gold": int(n_missing_gold),
    }

    # ---- label metrics ----
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        pr, rc, f1s, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_class = {}
        for i, lab in enumerate(labels):
            per_class[lab] = {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1s[i]),
                "support": int(sup[i]),
            }

        report["label"] = {
            "n_scored": int(len(y_true)),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "labels_order": labels,
            "confusion_matrix": cm,
            "per_class": per_class,
            "pred_counts": dict(pred_counts),
            "true_counts": dict(true_counts),
        }
    else:
        report["label"] = {
            "n_scored": 0,
            "accuracy": None,
            "macro_f1": None,
            "labels_order": labels,
            "confusion_matrix": None,
        }

    # ---- evidence metrics ----
    if ev_scored > 0:
        report["evidence"] = {
            "n_scored": int(ev_scored),
            "mean_precision": float(ev_prec_sum / ev_scored),
            "mean_recall": float(ev_rec_sum / ev_scored),
            "mean_f1": float(ev_f1_sum / ev_scored),

            "avg_predicted_sentences": float(total_pred_ev / ev_scored),
            "avg_gold_sentences": float(total_gold_ev / ev_scored),
            "avg_candidate_sentences": float(total_cand_ev / ev_scored),

            # Demoted: overall empty (kept for debugging, don’t use as headline)
            "pct_pred_empty_overall": float(pred_empty_overall / ev_scored),

            # NEW: much higher-fidelity diagnostics
            "pct_empty_when_pred_non_nei": (float(empty_when_pred_non_nei / n_pred_non_nei) if n_pred_non_nei > 0 else None),
            "pct_nonempty_when_pred_nei": (float(nonempty_when_pred_nei / n_pred_nei) if n_pred_nei > 0 else None),
            "n_pred_non_nei": int(n_pred_non_nei),
            "n_pred_nei": int(n_pred_nei),

            "pct_cite_all_unnecessary": float(cite_all_unnecessary / ev_scored),
            "pct_pred_shotgun": float(shotgun / ev_scored),

            "correct_label_wrong_evidence": int(correct_label_wrong_evidence),
            "wrong_label_right_evidence": int(wrong_label_right_evidence),

            "shotgun_thresholds": {
                "prec_cutoff": PREC_CUTOFF,
                "extra_abs": EXTRA_ABS,
                "extra_mult": EXTRA_MULT,
            },
        }
    else:
        report["evidence"] = {
            "n_scored": 0,
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
        }

    print(json.dumps(report, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
