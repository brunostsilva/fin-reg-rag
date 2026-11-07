import pytest
from pydantic import ValidationError

from finregrag.config import Settings

_VARIABLES = ["LLM_PROVIDER", "EMBEDDINGS_PROVIDER", "VECTOR_STORE", "TOP_K", "CHUNK_SIZE", "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Runs each test in an empty directory (no .env) without the variables under test."""
    monkeypatch.chdir(tmp_path)
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_run_fully_offline() -> None:
    settings = Settings()

    assert (settings.llm_provider, settings.embeddings_provider, settings.vector_store) == ("fake", "local", "faiss")
    assert settings.openai_api_key == ""
    assert settings.top_k == 4


def test_environment_variables_override_the_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("TOP_K", "7")

    settings = Settings()

    assert (settings.llm_provider, settings.vector_store, settings.top_k) == ("openai", "pgvector", 7)


def test_dotenv_is_read_and_the_environment_takes_precedence(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("TOP_K=9\nCHUNK_SIZE=500\nSOMETHING_UNRELATED=1\n", encoding="utf-8")
    monkeypatch.setenv("CHUNK_SIZE", "300")

    settings = Settings()

    assert settings.top_k == 9  # from .env
    assert settings.chunk_size == 300  # the environment wins over .env
    # unknown keys in .env are ignored (extra="ignore") instead of failing


def test_invalid_numbers_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TOP_K", "many")

    with pytest.raises(ValidationError):
        Settings()
