from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


class VectorStore(Protocol):
    def add(self, embeddings: np.ndarray, texts: list[str], sources: list[str], chunk_indices: list[int]) -> None: ...

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]: ...

    def delete_source(self, source: str) -> int: ...

    def count(self) -> int: ...
