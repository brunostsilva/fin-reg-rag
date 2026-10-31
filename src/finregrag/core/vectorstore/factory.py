from __future__ import annotations

from finregrag.core.vectorstore.base import VectorStore


def build_vector_store(
    kind: str,
    *,
    dimension: int,
    faiss_index_path: str = "",
    postgres_dsn: str = "",
) -> VectorStore:
    if kind == "faiss":
        from finregrag.core.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(dimension=dimension, index_path=faiss_index_path)
    if kind == "pgvector":
        from finregrag.core.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(dsn=postgres_dsn, dimension=dimension)
    raise ValueError(f"Unknown vector store: {kind!r}")
