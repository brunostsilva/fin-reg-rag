from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")

_OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbeddingsProvider(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalHashEmbeddings:
    """Deterministic, offline embedding provider.

    Uses the hashing trick (token -> bucket via md5) with L2-normalized term
    counts, so no model download or network/API call is required. It is not a
    substitute for a learned embedding model in production, but it keeps the
    rest of the RAG pipeline (chunking, retrieval, generation) fully testable
    and runnable without any external dependency or API key.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectorize(t) for t in texts]) if texts else np.empty((0, self.dimension))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)


class OpenAIEmbeddings:
    """Thin wrapper around the OpenAI embeddings API via langchain-openai.

    Imported lazily so the dependency is only required when this provider is
    actually selected (LLM_PROVIDER/EMBEDDINGS_PROVIDER configuration).
    """

    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

        self._client = LangchainOpenAIEmbeddings(model=model, api_key=api_key)
        known = _OPENAI_EMBEDDING_DIMENSIONS.get(model)
        self.dimension = known if known is not None else len(self._client.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension))
        return np.array(self._client.embed_documents(texts), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._client.embed_query(text), dtype=np.float32)


def build_embeddings_provider(provider: str, *, model: str = "", api_key: str = "") -> EmbeddingsProvider:
    if provider == "local":
        return LocalHashEmbeddings()
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=api_key)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")
