import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Make it easy to import from src/ without packaging yet
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from ollama_client import call_ollama_generate, OllamaResult  # noqa: E402


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


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_completed_ids(output_path: Path) -> set[str]:
    """
    Allows resume: if the output file already has results, we skip IDs already processed.
    """
    if not output_path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(output_path):
        rid = row.get("id")
        if isinstance(rid, str):
            done.add(rid)
    return done


def normalize_label(label: str) -> str:
    """
    Normalize common label variants into SUPPORTS/REFUTES/NEI.
    Keeps evaluation robust to small output formatting differences.
    """
    if not isinstance(label, str):
        return "INVALID"
    x = label.strip().upper()
    mapping = {
        "SUPPORTS": "SUPPORTS",
        "SUPPORT": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "REFUTES": "REFUTES",
        "REFUTE": "REFUTES",
        "REFUTED": "REFUTES",
        "NEI": "NEI",
        "NOT_ENOUGH_INFO": "NEI",
        "NOT ENOUGH INFO": "NEI",
        "INSUFFICIENT": "NEI",
        "INSUFFICIENT_EVIDENCE": "NEI",
    }
    return mapping.get(x, x)


def build_prompt_from_row(row: Dict[str, Any]) -> str:
    """
    We support two input styles:
      A) row contains a prebuilt 'prompt' string (we just send it).
      B) row contains 'claim' and 'evidence' list -> we format a prompt.

    Evidence format expected for style (B):
      evidence = [{"sent_id": 1, "sentence": "..."}, ...]
    """
    if "prompt" in row and isinstance(row["prompt"], str) and row["prompt"].strip():
        return row["prompt"].strip()

    claim = row.get("claim")
    evidence = row.get("evidence")

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Row missing 'claim' (string) and no 'prompt' provided.")

    if not isinstance(evidence, list) or len(evidence) == 0:
        raise ValueError("Row missing 'evidence' (non-empty list) and no 'prompt' provided.")

    # Build numbered evidence block
    lines = []
    for item in evidence:
        sent_id = item.get("sent_id")
        sent = item.get("sentence")
        if not isinstance(sent_id, int) or not isinstance(sent, str):
            raise ValueError("Evidence items must have {'sent_id': int, 'sentence': str}.")
        lines.append(f'{sent_id}) "{sent.strip()}"')

    evidence_block = "\n".join(lines)

    prompt = f"""Return ONLY valid JSON with keys: label, evidence_sent_ids, rationale.
label must be one of: SUPPORTS, REFUTES, NEI.
evidence_sent_ids must be a list of integers (can be empty).
rationale must be <= 2 sentences.

Claim: "{claim.strip()}"
Evidence sentences (numbered):
{evidence_block}

Decide SUPPORTS/REFUTES/NEI and cite which sentence numbers support your decision.
"""
    return prompt


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def result_row(
    input_row: Dict[str, Any],
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    res: OllamaResult,
    elapsed_s: float,
) -> Dict[str, Any]:
    """
    Produce a single JSON-serializable output record (one line in runs/*.jsonl).
    Includes raw output + validated fields + error info for debugging.
    """
    out: Dict[str, Any] = {
        "id": input_row.get("id"),
        "model": model,
        "host": host,
        "temperature": temperature,
        "prompt_sha256": sha256_text(prompt),
        "elapsed_s": elapsed_s,
        "raw_text": res.raw_text,
        "error": res.error,
    }

    # Always store parsed JSON if we got it
    out["parsed_json"] = res.parsed_json

    # If schema validated, store a clean canonical version
    if res.validated is not None:
        canon = res.validated.model_dump()
        # Normalize label for downstream metrics robustness
        canon["label"] = normalize_label(canon.get("label", ""))
        out["validated"] = canon
    else:
        out["validated"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL (appended).")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="If >0, process only first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose IDs are already in output.")
    parser.add_argument("--print_every", type=int, default=10, help="Log progress every N rows.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    done_ids = load_completed_ids(output_path) if args.resume else set()

    n_total = 0
    n_skipped = 0
    n_ok = 0
    n_err = 0

    for row in read_jsonl(input_path):
        n_total += 1
        if args.limit and n_total > args.limit:
            break

        rid = row.get("id")
        if args.resume and isinstance(rid, str) and rid in done_ids:
            n_skipped += 1
            continue

        try:
            prompt = build_prompt_from_row(row)
        except Exception as e:
            # Prompt construction error: write it out as an error record
            out = {
                "id": rid,
                "model": args.model,
                "host": args.host,
                "temperature": args.temperature,
                "prompt_sha256": None,
                "elapsed_s": 0.0,
                "raw_text": "",
                "parsed_json": None,
                "validated": None,
                "error": f"Prompt build error: {e}",
            }
            append_jsonl(output_path, out)
            n_err += 1
            continue

        t0 = time.time()
        res = call_ollama_generate(
            host=args.host,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0

        out = result_row(
            input_row=row,
            prompt=prompt,
            model=args.model,
            host=args.host,
            temperature=args.temperature,
            res=res,
            elapsed_s=elapsed,
        )
        append_jsonl(output_path, out)

        if res.error:
            n_err += 1
        else:
            n_ok += 1

        if args.print_every and (n_total % args.print_every == 0):
            print(
                f"[{n_total}] ok={n_ok} err={n_err} skipped={n_skipped} -> {output_path}"
            )

    print(f"Done. processed={n_total} ok={n_ok} err={n_err} skipped={n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
