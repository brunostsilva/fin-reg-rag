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
            f"[Fonte: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
        )
        prompt = f"Contexto:\n{context}\n\nPergunta: {question}"
        response = self._client.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return response.content
