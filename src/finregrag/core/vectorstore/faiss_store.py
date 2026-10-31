from __future__ import annotations

import json
import os

import faiss
import numpy as np

from finregrag.core.vectorstore.base import RetrievedChunk


class FaissVectorStore:
    """Local, file-backed vector store using FAISS inner-product search.

    Embeddings are expected to already be L2-normalized by the embeddings
    provider, so inner product is equivalent to cosine similarity.
    """

    def __init__(self, dimension: int, index_path: str) -> None:
        self._dimension = dimension
        self._index_path = index_path
        self._metadata_path = f"{index_path}.meta.json"
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._metadata_path):
            index = faiss.read_index(self._index_path)
            if index.d != self._dimension:
                raise ValueError(
                    f"FAISS index at {self._index_path!r} has dimension {index.d}, but the configured "
                    f"embeddings provider produces dimension {self._dimension}. Delete the index "
                    "(and its .meta.json) and re-ingest, or switch back to the original provider."
                )
            self._index = index
            with open(self._metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)

    def persist(self) -> None:
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

    def add(self, embeddings: np.ndarray, texts: list[str], sources: list[str], chunk_indices: list[int]) -> None:
        if len(texts) == 0:
            return
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        for text, source, chunk_index in zip(texts, sources, chunk_indices):
            self._metadata.append({"text": text, "source": source, "chunk_index": chunk_index})

    def delete_source(self, source: str) -> int:
        positions = [i for i, meta in enumerate(self._metadata) if meta["source"] == source]
        if not positions:
            return 0
        # Flat indexes compact on removal while preserving order, so the
        # metadata list stays aligned with index positions.
        self._index.remove_ids(np.array(positions, dtype=np.int64))
        removed = set(positions)
        self._metadata = [meta for i, meta in enumerate(self._metadata) if i not in removed]
        return len(positions)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []
        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    score=float(score),
                )
            )
        return results

    def count(self) -> int:
        return self._index.ntotal
