import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

RAW_LABEL_TO_3WAY = {
    "SUPPORT": "SUPPORTS",
    "CONTRADICT": "REFUTES",
    "REFUTE": "REFUTES",
    "NOT_ENOUGH_INFO": "NEI",
    "NOINFO": "NEI",
    "NEI": "NEI",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

def extract_tar_gz(tar_path: Path, out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists() and not force:
        return
    # Clear only marker; keep files if already extracted unless force
    if force and out_dir.exists():
        # Lightweight “force”: re-extract over existing
        pass
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    marker.write_text("ok", encoding="utf-8")

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_corpus(corpus_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Returns: doc_id -> {title: str, abstract: List[str], structured: bool}
    """
    corpus: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(corpus_path):
        doc_id = int(row["doc_id"])
        corpus[doc_id] = {
            "title": row.get("title", ""),
            "abstract": row.get("abstract", []),
            "structured": bool(row.get("structured", False)),
        }
    return corpus

def pick_label_from_evidence(evidence: Dict[str, Any]) -> Optional[str]:
    """
    In SciFact, evidence is typically:
      evidence = { "<doc_id>": [ {"label": "SUPPORT|CONTRADICT", "sentences": [..]}, ...], ... }
    We assume labels are consistent; if not, we choose a majority.
    """
    labels: List[str] = []
    for doc_id, ev_sets in evidence.items():
        for ev in ev_sets:
            lbl = ev.get("label")
            if lbl:
                labels.append(lbl)
    if not labels:
        return None
    # Majority vote
    counts: Dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def choose_doc_and_sentences(
    evidence: Dict[str, Any]
) -> Tuple[Optional[int], List[int]]:
    """
    Chooses ONE doc_id and unions all sentence ids across all evidence sets for that doc.
    Returns (doc_id_int, sentence_ids_zero_based)
    """
    if not evidence:
        return None, []
    # deterministic: first doc_id key (sorted)
    doc_keys = sorted(evidence.keys(), key=lambda x: int(x))
    doc_id = int(doc_keys[0])
    sent_ids: set[int] = set()
    for ev in evidence[doc_keys[0]]:
        for s in ev.get("sentences", []):
            sent_ids.add(int(s))
    return doc_id, sorted(sent_ids)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/scifact_oracle_dev_200.jsonl")
    ap.add_argument("--cache_dir", type=str, default="data/raw/scifact")
    ap.add_argument("--force_redownload", action="store_true")
    ap.add_argument("--force_reextract", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    tar_path = cache_dir / "data.tar.gz"
    extract_dir = cache_dir / "extracted"

    # 1) Download + extract
    download(SCIFACT_URL, tar_path, force=args.force_redownload)
    extract_tar_gz(tar_path, extract_dir, force=args.force_reextract)

    # Files inside tar are typically under "data/"
    data_dir = extract_dir / "data"
    corpus_path = data_dir / "corpus.jsonl"
    claims_path = data_dir / f"claims_{args.split}.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus.jsonl at {corpus_path}")
    if not claims_path.exists():
        raise FileNotFoundError(f"Missing claims file at {claims_path}")

    # 2) Load corpus
    corpus = load_corpus(corpus_path)
    corpus_doc_ids = list(corpus.keys())

    # 3) Build examples
    rng = random.Random(args.seed)
    examples: List[Dict[str, Any]] = []

    for row in read_jsonl(claims_path):
        claim_id = row.get("id")
        claim_text = row.get("claim", "").strip()
        cited_doc_ids = row.get("cited_doc_ids", []) or []
        evidence = row.get("evidence", {}) or {}

        # Determine label
        raw_label = row.get("label")  # some releases include this
        if not raw_label:
            raw_label = pick_label_from_evidence(evidence)
        if not raw_label:
            # If no evidence and no label field, treat as NOINFO/NEI
            raw_label = "NOINFO"

        gold_label = RAW_LABEL_TO_3WAY.get(raw_label)
        if gold_label is None:
            # If your release uses something slightly different, we fail loudly.
            raise ValueError(f"Unknown raw label: {raw_label}")

        # Choose doc + evidence sentences
        doc_id, sent_ids_0 = choose_doc_and_sentences(evidence)

        # For NEI, we still want *some* abstract to show the model (oracle “given doc”),
        # otherwise the prompt has no evidence text.
        if doc_id is None:
            if cited_doc_ids:
                doc_id = int(cited_doc_ids[0])
            else:
                # fallback: pick a random corpus doc
                doc_id = int(rng.choice(corpus_doc_ids))

        doc = corpus.get(int(doc_id))
        if not doc:
            # if doc_id not in corpus, skip (rare)
            continue

        abstract_sents: List[str] = doc["abstract"]

        # Convert to 1-based sentence IDs for your prompt/metrics convention
        evidence_list = [{"sent_id": i + 1, "sentence": s} for i, s in enumerate(abstract_sents)]
        gold_evidence_1 = [s + 1 for s in sent_ids_0] if gold_label != "NEI" else []

        ex = {
            "id": f"scifact_{args.split}_{claim_id}_{doc_id}",
            "claim": claim_text,
            "gold_label": gold_label,
            "gold_evidence_sent_ids": gold_evidence_1,
            "doc_id": int(doc_id),
            "title": doc.get("title", ""),
            "evidence": evidence_list,
        }
        examples.append(ex)

    # 4) Sample and write JSONL
    rng.shuffle(examples)
    examples = examples[: args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(json.dumps(
        {
            "out": str(out_path),
            "split": args.split,
            "n_written": len(examples),
            "tar_sha256": sha256_file(tar_path),
        },
        indent=2
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
