import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

# Make it easy to import from src/ without packaging yet
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from run_judge import run_judge, ModelSpec  # noqa: E402
from ollama_client import OllamaResult  # noqa: E402


DEFAULT_TEMPLATE = """Return ONLY valid JSON with keys: label, evidence_sent_ids, rationale.
label must be one of: SUPPORTS, REFUTES, NEI.
evidence_sent_ids must be a list of integers (can be empty).
rationale must be a single STRING (not a list/array) and <= 2 sentences.

Claim: "{claim}"
Evidence sentences (numbered):
{evidence_block}

Decide SUPPORTS/REFUTES/NEI and cite which sentence numbers support your decision.
"""

def load_prompt_template(prompt_file: str | None) -> str:
    if not prompt_file:
        return DEFAULT_TEMPLATE
    return Path(prompt_file).read_text(encoding="utf-8")

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
    if not output_path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(output_path):
        rid = row.get("id")
        err = row.get("error")
        if isinstance(rid, str) and not err and row.get("validated") is not None:
            done.add(rid)
    return done

def normalize_label(label: str) -> str:
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

def build_prompt_from_row(row: dict, template: str) -> str:
    claim = (row.get("claim") or "").strip()
    evidence = row.get("evidence") or []
    evidence_lines = []
    for e in evidence:
        sid = e.get("sent_id")
        sent = (e.get("sentence") or "").strip()
        evidence_lines.append(f"{sid}) {sent}")
    evidence_block = "\n".join(evidence_lines)
    return template.format(claim=claim, evidence_block=evidence_block)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def result_row(
    input_row: Dict[str, Any],
    prompt: str,
    backend: str,
    model: str,
    host: str,
    temperature: float,
    res: OllamaResult,
    elapsed_s: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": input_row.get("id"),
        "backend": backend,
        "model": model,
        "host": host,
        "temperature": temperature,
        "prompt_sha256": sha256_text(prompt),
        "elapsed_s": elapsed_s,
        "raw_text": res.raw_text,
        "error": res.error,
        "parsed_json": res.parsed_json,
    }

    if res.validated is not None:
        canon = res.validated.model_dump()
        canon["label"] = normalize_label(canon.get("label", ""))
        out["validated"] = canon
    else:
        out["validated"] = None

    return out

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL (appended).")
    parser.add_argument("--backend", type=str, default="ollama", help="ollama | openai (default: ollama)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="If >0, process only first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose IDs are already in output.")
    parser.add_argument("--print_every", type=int, default=10, help="Log progress every N rows.")
    parser.add_argument("--prompt_file", type=str, default=None)
    args = parser.parse_args()

    template = load_prompt_template(args.prompt_file)
    input_path = Path(args.input)
    output_path = Path(args.output)

    done_ids = load_completed_ids(output_path) if args.resume else set()

    spec = ModelSpec(
        backend=args.backend,
        model=args.model,
        host=args.host,
        temperature=args.temperature,
    )

    n_total = 0
    n_skipped = 0
    n_ok = 0
    n_err = 0

    host_field = args.host if args.backend == "ollama" else "openai"

    for row in read_jsonl(input_path):
        n_total += 1
        if args.limit and n_total > args.limit:
            break

        rid = row.get("id")
        if args.resume and isinstance(rid, str) and rid in done_ids:
            n_skipped += 1
            continue

        try:
            prompt = build_prompt_from_row(row, template)
        except Exception as e:
            append_jsonl(output_path, {
                "id": rid,
                "backend": args.backend,
                "model": args.model,
                "host": host_field,
                "temperature": args.temperature,
                "prompt_sha256": None,
                "elapsed_s": 0.0,
                "raw_text": "",
                "parsed_json": None,
                "validated": None,
                "error": f"Prompt build error: {e}",
            })
            n_err += 1
            continue

        t0 = time.time()
        res = run_judge(spec, prompt=prompt)
        elapsed = time.time() - t0

        out = result_row(
            input_row=row,
            prompt=prompt,
            backend=args.backend,
            model=args.model,
            host=host_field,
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
            print(f"[{n_total}] ok={n_ok} err={n_err} skipped={n_skipped} -> {output_path}")

    print(f"Done. processed={n_total} ok={n_ok} err={n_err} skipped={n_skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
