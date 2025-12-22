import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field, ValidationError


# --- 1) Define the JSON schema you want the model to output ---
class JudgeResponse(BaseModel):
    label: str = Field(..., description="One of: SUPPORTS, REFUTES, NEI")
    evidence_sent_ids: list[int] = Field(default_factory=list)
    rationale: str = Field(..., description="Short explanation (<=2 sentences)")


# --- 2) Helpers ---
def _extract_first_json_object(text: str) -> Optional[str]:
    """
    If the model returns extra text, try to grab the first {...} block.
    Very simple heuristic: first '{' to last '}'.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


@dataclass
class OllamaResult:
    raw_text: str
    parsed_json: Optional[dict[str, Any]]
    validated: Optional[JudgeResponse]
    error: Optional[str]


def call_ollama_generate(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout_s: int = 120,
) -> OllamaResult:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
    except Exception as e:
        return OllamaResult(raw_text="", parsed_json=None, validated=None, error=f"HTTP error: {e}")

    try:
        raw_text = r.json()["response"]
    except Exception as e:
        return OllamaResult(raw_text=r.text, parsed_json=None, validated=None, error=f"Bad response JSON: {e}")

    json_str = _extract_first_json_object(raw_text)
    if json_str is None:
        return OllamaResult(raw_text=raw_text, parsed_json=None, validated=None, error="No JSON object found in output")

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        return OllamaResult(raw_text=raw_text, parsed_json=None, validated=None, error=f"JSON parse failed: {e}")

    try:
        validated = JudgeResponse.model_validate(parsed)
    except ValidationError as e:
        return OllamaResult(raw_text=raw_text, parsed_json=parsed, validated=None, error=f"Schema validation failed: {e}")

    return OllamaResult(raw_text=raw_text, parsed_json=parsed, validated=validated, error=None)



def build_default_prompt() -> str:
    # Minimal prompt to test the pipeline. We'll replace this with SciFact later.
    return """Return ONLY JSON with keys: label, evidence_sent_ids, rationale.
label must be one of: SUPPORTS, REFUTES, NEI.
evidence_sent_ids must be a list of integers (can be empty).
rationale must be <= 2 sentences.

Claim: "Vitamin C prevents the common cold."
Evidence sentences (numbered):
1) "Multiple randomized trials show vitamin C does not prevent colds in the general population."
2) "Vitamin C may slightly reduce cold duration in some subgroups."

Decide SUPPORTS/REFUTES/NEI and cite which sentence numbers support your decision.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None, help="Inline prompt string. If omitted, uses default.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a text file with the prompt.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--print_raw", action="store_true", help="Also print raw model output.")
    args = parser.parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file, "r", encoding="utf-8").read()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = build_default_prompt()

    res = call_ollama_generate(
        host=args.host,
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
    )

    if args.print_raw:
        print("\n--- RAW OUTPUT ---\n")
        print(res.raw_text)

    if res.error:
        print("\n--- ERROR ---\n")
        print(res.error)
        return 1

    # Print validated JSON (canonical)
    print(json.dumps(res.validated.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
