from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "fake"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"

    embeddings_provider: str = "local"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store: str = "faiss"
    faiss_index_path: str = "./data/faiss_index"
    postgres_dsn: str = "postgresql://finregrag:finregrag@localhost:5432/finregrag"

    docs_dir: str = "./data/sample_docs"
    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 4


def get_settings() -> Settings:
    return Settings()
