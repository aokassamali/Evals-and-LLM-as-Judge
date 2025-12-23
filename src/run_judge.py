from dataclasses import dataclass
from typing import Optional

from ollama_client import call_ollama_generate, OllamaResult, JudgeResponse

@dataclass
class ModelSpec:
    backend: str          # "ollama" | "openai"
    model: str            # e.g. "llama3.1:8b" or "gpt-5-mini"
    host: str = "http://localhost:11434"
    temperature: float = 0.0

def run_judge(spec: ModelSpec, prompt: str) -> OllamaResult:
    if spec.backend == "ollama":
        return call_ollama_generate(
            host=spec.host,
            model=spec.model,
            prompt=prompt,
            temperature=spec.temperature,
        )
    if spec.backend == "openai":
        from openai_backend import call_openai_judge
        return call_openai_judge(model=spec.model, prompt=prompt, temperature=spec.temperature)

    return OllamaResult(raw_text="", parsed_json=None, validated=None, error=f"Unknown backend: {spec.backend}")
