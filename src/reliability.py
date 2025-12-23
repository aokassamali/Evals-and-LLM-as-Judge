import argparse
import itertools
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


DEFAULT_LABELS = ["SUPPORTS", "REFUTES", "NEI"]


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_run_labels(path: str) -> Tuple[Dict[str, str], int]:
    """
    Returns:
      labels_by_id: {id -> label}
      n_errors: count of rows with non-null "error"
    """
    rows = read_jsonl(path)
    labels_by_id: Dict[str, str] = {}
    n_errors = 0

    for r in rows:
        if r.get("error"):
            n_errors += 1
            continue

        rid = r.get("id")
        if not rid:
            continue

        # Prefer validated; fall back to parsed_json if needed
        payload = r.get("validated") or r.get("parsed_json") or {}
        label = payload.get("label")
        if not label:
            continue

        labels_by_id[rid] = label

    return labels_by_id, n_errors


def percent_agreement(a: Dict[str, str], b: Dict[str, str], ids: List[str]) -> float:
    if not ids:
        return 0.0
    agree = sum(1 for i in ids if a[i] == b[i])
    return agree / len(ids)


def cohen_kappa(
    a: Dict[str, str], b: Dict[str, str], ids: List[str], labels: List[str]
) -> float:
    """
    Cohen's kappa for two raters over a shared set of item IDs.
    """
    n = len(ids)
    if n == 0:
        return 0.0

    po = percent_agreement(a, b, ids)

    ca = Counter(a[i] for i in ids)
    cb = Counter(b[i] for i in ids)

    pe = 0.0
    for lab in labels:
        pe += (ca.get(lab, 0) / n) * (cb.get(lab, 0) / n)

    if pe >= 0.999999:
        # Avoid division weirdness when marginals are degenerate
        return 1.0 if po >= 0.999999 else 0.0

    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: List[List[str]], labels: List[str]) -> float:
    """
    Fleiss' kappa for N items, k raters, categorical labels.
    ratings: list of length N, each element is list of length k labels.
    """
    N = len(ratings)
    if N == 0:
        return 0.0
    k = len(ratings[0])
    if k < 2:
        return 0.0

    # Count category assignments per item
    n_ij = []
    for item_ratings in ratings:
        c = Counter(item_ratings)
        n_ij.append([c.get(lab, 0) for lab in labels])

    # P_i agreement per item
    P_i = []
    for counts in n_ij:
        s = sum(n * (n - 1) for n in counts)
        P_i.append(s / (k * (k - 1)))

    P_bar = sum(P_i) / N

    # Category proportions across all assignments
    total_assignments = N * k
    p_j = []
    for j, lab in enumerate(labels):
        p = sum(n_ij[i][j] for i in range(N)) / total_assignments
        p_j.append(p)

    P_e = sum(p * p for p in p_j)

    if P_e >= 0.999999:
        return 1.0 if P_bar >= 0.999999 else 0.0

    return (P_bar - P_e) / (1.0 - P_e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to run JSONL files (from run_batch.py).",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional names for runs (same order as --runs).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=DEFAULT_LABELS,
        help="Label set and order (default: SUPPORTS REFUTES NEI).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="How many disagreement examples to print.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path to write summary JSON.",
    )
    args = parser.parse_args()

    run_paths: List[str] = args.runs
    run_names: List[str] = args.names if args.names else [f"run{i+1}" for i in range(len(run_paths))]
    if len(run_names) != len(run_paths):
        raise RuntimeError("If provided, --names must have the same length as --runs.")

    labels = args.labels

    runs: List[Dict[str, str]] = []
    errors_by_run = {}
    for name, path in zip(run_names, run_paths):
        lab, n_err = load_run_labels(path)
        runs.append(lab)
        errors_by_run[name] = n_err

    # Shared IDs across all runs
    shared_ids = set(runs[0].keys())
    for r in runs[1:]:
        shared_ids &= set(r.keys())
    shared_ids = sorted(shared_ids)

    summary = {
        "n_runs": len(runs),
        "run_names": run_names,
        "labels_order": labels,
        "errors_by_run": errors_by_run,
        "n_shared_ids_all_runs": len(shared_ids),
    }

    # Pairwise agreement / kappa
    pairwise = []
    for (i, (name_a, a)), (j, (name_b, b)) in itertools.combinations(enumerate(zip(run_names, runs)), 2):
        ids = sorted(set(a.keys()) & set(b.keys()))
        pa = percent_agreement(a, b, ids)
        kappa = cohen_kappa(a, b, ids, labels)
        pairwise.append(
            {
                "a": name_a,
                "b": name_b,
                "n_shared": len(ids),
                "percent_agreement": pa,
                "cohen_kappa": kappa,
            }
        )
    summary["pairwise"] = pairwise

    # Fleiss kappa across all runs on intersection
    if len(shared_ids) > 0 and len(runs) >= 2:
        ratings_matrix = []
        for rid in shared_ids:
            ratings_matrix.append([r[rid] for r in runs])
        summary["fleiss_kappa_all_runs"] = fleiss_kappa(ratings_matrix, labels)
    else:
        summary["fleiss_kappa_all_runs"] = None

    # Disagreements (IDs where not all runs agree)
    disagreements = []
    for rid in shared_ids:
        labs = [r[rid] for r in runs]
        if len(set(labs)) > 1:
            disagreements.append({"id": rid, "labels": {name: lab for name, lab in zip(run_names, labs)}})

    summary["n_disagreements_all_runs"] = len(disagreements)
    summary["sample_disagreements"] = disagreements[: args.top_k]

    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
