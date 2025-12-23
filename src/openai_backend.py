from openai import OpenAI

from ollama_client import JudgeResponse, OllamaResult

client = OpenAI()

def call_openai_judge(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    reasoning_effort: str = "none",
    max_output_tokens: int = 256,
    store: bool = False,
) -> OllamaResult:
    """
    OpenAI judge call using Responses API + Structured Outputs (Pydantic schema).

    Note: many GPT-5 family configurations reject non-default temperature.
    We keep temperature in the signature for interface compatibility, but we
    only send it for non-gpt-5* models.
    """
    try:
        kwargs = dict(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=JudgeResponse,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
            store=store,
        )
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = temperature

        resp = client.responses.parse(**kwargs)
        parsed: JudgeResponse = resp.output_parsed

        raw_text = getattr(resp, "output_text", None) or parsed.model_dump_json()

        return OllamaResult(
            raw_text=raw_text,
            parsed_json=parsed.model_dump(),
            validated=parsed,
            error=None,
        )
    except Exception as e:
        return OllamaResult(raw_text="", parsed_json=None, validated=None, error=str(e))
