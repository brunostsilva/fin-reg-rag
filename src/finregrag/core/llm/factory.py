from __future__ import annotations

from finregrag.core.llm.base import LLMProvider


def build_llm_provider(provider: str, *, model: str = "", api_key: str = "") -> LLMProvider:
    if provider == "fake":
        from finregrag.core.llm.fake_llm import FakeExtractiveLLM

        return FakeExtractiveLLM()
    if provider == "openai":
        from finregrag.core.llm.openai_llm import OpenAIChatLLM

        return OpenAIChatLLM(model=model, api_key=api_key)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
