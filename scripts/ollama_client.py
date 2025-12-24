import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Optional, Literal, Tuple
import time

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator


# --- 1) Define the JSON schema you want the model to output ---
class JudgeResponse(BaseModel):
    label: Literal["SUPPORTS", "REFUTES", "NEI"]
    evidence_sent_ids: list[int] = Field(default_factory=list)
    rationale: str = Field(..., description="Short explanation (<=2 sentences)")

    @field_validator("evidence_sent_ids", mode="before")
    @classmethod
    def _coerce_evidence_ids(cls, v: Any) -> Any:
        # Allow ["1","2"] or "1,2" etc.
        if v is None:
            return []
        if isinstance(v, list):
            out = []
            for x in v:
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.strip().isdigit():
                    out.append(int(x.strip()))
            return out
        if isinstance(v, str):
            # "1, 2" -> [1,2]
            parts = [p.strip() for p in v.replace("[", "").replace("]", "").split(",")]
            out = [int(p) for p in parts if p.isdigit()]
            return out
        return v

    @field_validator("rationale", mode="before")
    @classmethod
    def _coerce_rationale(cls, v: Any) -> Any:
        # If model outputs a list of strings, join into one string.
        if isinstance(v, list):
            pieces = [str(x).strip() for x in v if str(x).strip()]
            return " ".join(pieces)
        # Sometimes models return {"text": "..."} or similar
        if isinstance(v, dict) and "text" in v:
            return str(v["text"])
        return v


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

def _is_cloud_model(model: str) -> bool:
    # Ollama cloud tags commonly look like "deepseek-v3.2:cloud"
    return ":cloud" in model or model.endswith("-cloud")

def call_ollama_generate(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout_s: int = 120,
    max_retries: int = 3,
) -> OllamaResult:
    url = host.rstrip("/") + "/api/generate"

    schema = JudgeResponse.model_json_schema()

    # Ollama recommends also passing the schema as text in the prompt to "ground" the response. :contentReference[oaicite:4]{index=4}
    grounded_prompt = (
        prompt.rstrip()
        + "\n\nReturn ONLY valid JSON that matches this JSON Schema:\n"
        + json.dumps(schema, indent=2)
        + "\n"
    )

    is_cloud = _is_cloud_model(model)
    effective_timeout = max(timeout_s, 300) if is_cloud else timeout_s
    effective_retries = max(max_retries, 6) if is_cloud else max_retries

    payload = {
        "model": model,
        "prompt": grounded_prompt,
        "stream": False,
        "format": schema,  # structured output (JSON schema)
        "options": {
            "temperature": temperature,
            "num_predict": 200,
        },
    }

    if is_cloud:
        payload["keep_alive"] = "10m"

    last_body = None
    dropped_format = False

    for attempt in range(1, effective_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=effective_timeout)
            r.raise_for_status()
        except Exception as e:
            # retry transient issues (esp cloud)
            if attempt < effective_retries:
                time.sleep(min(2 ** attempt, 20) if is_cloud else 0.5 * attempt)
                continue
            return OllamaResult(raw_text="", parsed_json=None, validated=None, error=f"HTTP error: {e}")

        try:
            body = r.json()
            last_body = body
        except Exception as e:
            return OllamaResult(raw_text=r.text, parsed_json=None, validated=None, error=f"Bad response JSON: {e}")

        if isinstance(body, dict) and body.get("error"):
            return OllamaResult(
                raw_text="",
                parsed_json=None,
                validated=None,
                error=f"Ollama error field: {body.get('error')} (done_reason={body.get('done_reason')})",
            )

        # Primary: normal field
        raw_text = (body.get("response") or "").strip()

        # If response is empty, some thinking-capable models may still emit a `thinking` field. :contentReference[oaicite:5]{index=5}
        # We DO NOT want to print the full thinking trace. We only try to extract a JSON object from it.
        if not raw_text:
            thinking = body.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                json_str = _extract_first_json_object(thinking)
                if json_str:
                    raw_text = json_str.strip()

        if raw_text:
            json_str = _extract_first_json_object(raw_text) or raw_text
            try:
                parsed = json.loads(json_str)
            except Exception as e:
                return OllamaResult(raw_text=raw_text, parsed_json=None, validated=None, error=f"JSON parse failed: {e}")

            try:
                validated = JudgeResponse.model_validate(parsed)
            except ValidationError as e:
                return OllamaResult(raw_text=raw_text, parsed_json=parsed, validated=None, error=f"Schema validation failed: {e}")

            return OllamaResult(raw_text=raw_text, parsed_json=parsed, validated=validated, error=None)

        # Still empty: fallback — retry without `format` once (format+schema can trigger empty responses). :contentReference[oaicite:6]{index=6}
        if attempt < effective_retries:
            if (not dropped_format) and ("format" in payload):
                payload = dict(payload)
                payload.pop("format", None)
                dropped_format = True
            time.sleep(min(2 ** attempt, 20) if is_cloud else 0.5 * attempt)
            continue

        keys = list(last_body.keys()) if isinstance(last_body, dict) else None
        done = last_body.get("done") if isinstance(last_body, dict) else None
        done_reason = last_body.get("done_reason") if isinstance(last_body, dict) else None
        return OllamaResult(
            raw_text="",
            parsed_json=None,
            validated=None,
            error=f"Empty response from model. keys={keys} done={done} done_reason={done_reason}",
        )


        # ---- Empty response handling ----
        # This can happen intermittently with JSON/schema constrained output in some setups/models. :contentReference[oaicite:5]{index=5}
        # Fallback: if we were using `format` and got an empty response, try again WITHOUT `format`.
        if attempt < max_retries:
            if "format" in payload:
                payload = dict(payload)
                payload.pop("format", None)  # remove schema constraint as fallback
                # Keep the grounded prompt so it still tends to emit JSON
            time.sleep(min(2 ** attempt, 20) if is_cloud else 0.5 * attempt)
            continue

        keys = list(last_body.keys()) if isinstance(last_body, dict) else None
        done = last_body.get("done") if isinstance(last_body, dict) else None
        done_reason = last_body.get("done_reason") if isinstance(last_body, dict) else None

        return OllamaResult(
            raw_text="",
            parsed_json=None,
            validated=None,
            error=(
                "Empty response from model. "
                f"keys={keys} done={done} done_reason={done_reason} last_http_error={last_http_error}"
            ),
        )



def build_default_prompt() -> str:
    # Minimal prompt to test the pipeline. We'll replace this with SciFact later.
    return """Return ONLY JSON with keys: label, evidence_sent_ids, rationale.
label must be one of: SUPPORTS, REFUTES, NEI.
evidence_sent_ids must be a list of integers (can be empty).
rationale must be a single STRING (not a list/array).

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
