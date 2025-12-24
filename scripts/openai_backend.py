from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from ollama_client import JudgeResponse, OllamaResult

client = OpenAI()


def call_openai_judge(model: str, prompt: str, temperature: float = 0.0) -> OllamaResult:
    """
    Uses OpenAI Responses API structured outputs via the SDK .responses.parse helper.
    Falls back gracefully if the model rejects `temperature`.
    """
    try:
        # First try with temperature (supported by Responses create; parse generally forwards args).
        try:
            resp = client.responses.parse(
                model=model,
                input=[{"role": "user", "content": prompt}],
                text_format=JudgeResponse,
                temperature=temperature,
            )
        except Exception as e:
            msg = str(e)
            # Some models/endpoints can reject temperature; retry without.
            if "temperature" in msg and ("Unsupported" in msg or "unsupported" in msg):
                resp = client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=JudgeResponse,
                )
            else:
                raise

        parsed: JudgeResponse = resp.output_parsed
        return OllamaResult(
            raw_text=json.dumps(parsed.model_dump(), ensure_ascii=False),
            parsed_json=parsed.model_dump(),
            validated=parsed,
            error=None,
        )
    except Exception as e:
        return OllamaResult(raw_text="", parsed_json=None, validated=None, error=str(e))
