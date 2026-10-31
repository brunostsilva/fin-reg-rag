from __future__ import annotations

from typing import Protocol

from finregrag.core.vectorstore.base import RetrievedChunk


class LLMProvider(Protocol):
    def generate_answer(self, question: str, context_chunks: list[RetrievedChunk]) -> str: ...
