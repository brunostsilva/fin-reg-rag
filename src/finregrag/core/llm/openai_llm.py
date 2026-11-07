from __future__ import annotations

from finregrag.core.vectorstore.base import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are an assistant specialized in financial and payments regulation. "
    "Answer ONLY based on the provided context passages. "
    "If the context is not sufficient to answer, say so explicitly. "
    "Always cite the source (document name) at the end of your answer."
)


class OpenAIChatLLM:
    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def generate_answer(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[Source: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self._client.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        # In langchain-core 1.x, `content` may be a list of content blocks instead of a string;
        # `text` always joins the text blocks into a plain string.
        return response.text
