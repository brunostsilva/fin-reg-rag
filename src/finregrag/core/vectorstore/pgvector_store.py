from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from finregrag.core.vectorstore.base import RetrievedChunk


class PgVectorStore:
    """Postgres + pgvector backed vector store, for production-style deployments.

    Requires the `pgvector` extension to be available in the target database
    (see docker-compose.yml, which uses the `pgvector/pgvector` image).
    """

    def __init__(self, dsn: str, dimension: int) -> None:
        self._dsn = dsn
        self._dimension = dimension
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        # DDL can't take bound parameters (psycopg 3 binds server-side), so the
        # dimension is rendered as a SQL literal instead.
        self._conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding VECTOR({dimension}) NOT NULL
                )
                """
            ).format(dimension=sql.Literal(int(dimension)))
        )

    def add(self, embeddings: np.ndarray, texts: list[str], sources: list[str], chunk_indices: list[int]) -> None:
        if len(texts) == 0:
            return
        with self._conn.cursor() as cur:
            for embedding, text, source, chunk_index in zip(embeddings, texts, sources, chunk_indices):
                cur.execute(
                    """
                    INSERT INTO document_chunks (text, source, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (text, source, chunk_index, embedding),
                )

    def delete_source(self, source: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE source = %s", (source,))
            return cur.rowcount

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, source, chunk_index, 1 - (embedding <=> %s) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(text=row[0], source=row[1], chunk_index=row[2], score=float(row[3]))
            for row in rows
        ]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            return cur.fetchone()[0]
