from __future__ import annotations

from dataclasses import dataclass

from finregrag.core.chunking import chunk_document
from finregrag.core.embeddings import EmbeddingsProvider
from finregrag.core.llm.base import LLMProvider
from finregrag.core.vectorstore.base import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]


class RagPipeline:
    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        vector_store: VectorStore,
        llm: LLMProvider,
        top_k: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k

    def ingest(self, text: str, source: str, chunk_size: int, chunk_overlap: int) -> int:
        """Index `text` under `source`, replacing any chunks previously indexed for it."""
        self._vector_store.delete_source(source)
        chunks = chunk_document(text, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return 0
        embeddings = self._embeddings.embed_documents([c.text for c in chunks])
        self._vector_store.add(
            embeddings=embeddings,
            texts=[c.text for c in chunks],
            sources=[c.source for c in chunks],
            chunk_indices=[c.chunk_index for c in chunks],
        )
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> RagAnswer:
        query_embedding = self._embeddings.embed_query(question)
        retrieved = self._vector_store.search(query_embedding, top_k=top_k or self._top_k)
        answer = self._llm.generate_answer(question, retrieved)
        return RagAnswer(answer=answer, sources=retrieved)

    def document_count(self) -> int:
        return self._vector_store.count()

    def persist(self) -> None:
        persist_fn = getattr(self._vector_store, "persist", None)
        if callable(persist_fn):
            persist_fn()
