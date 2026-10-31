from __future__ import annotations

from functools import lru_cache

from finregrag.config import Settings, get_settings
from finregrag.core.embeddings import build_embeddings_provider
from finregrag.core.llm.factory import build_llm_provider
from finregrag.core.rag_pipeline import RagPipeline
from finregrag.core.vectorstore.factory import build_vector_store


def build_pipeline(settings: Settings) -> RagPipeline:
    embeddings = build_embeddings_provider(
        settings.embeddings_provider,
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = build_vector_store(
        settings.vector_store,
        dimension=embeddings.dimension,
        faiss_index_path=settings.faiss_index_path,
        postgres_dsn=settings.postgres_dsn,
    )
    llm = build_llm_provider(
        settings.llm_provider,
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
    )
    return RagPipeline(embeddings=embeddings, vector_store=vector_store, llm=llm, top_k=settings.top_k)


@lru_cache
def get_pipeline() -> RagPipeline:
    return build_pipeline(get_settings())
