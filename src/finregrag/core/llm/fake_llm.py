from __future__ import annotations

from finregrag.core.vectorstore.base import RetrievedChunk


class FakeExtractiveLLM:
    """Offline, no-API-key fallback "generator".

    Instead of calling a real language model, it returns the single most
    relevant retrieved chunk verbatim, labeled with its source. This keeps
    the end-to-end pipeline runnable (and testable in CI) without any
    external dependency, while still demonstrating retrieval quality.
    """

    def generate_answer(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        if not context_chunks:
            return "I couldn't find any relevant passage in the document store to answer this question."
        best = context_chunks[0]
        return (
            f"[Extractive answer, no LLM configured] "
            f"Based on '{best.source}', the most relevant passage is:\n\n{best.text}"
        )
