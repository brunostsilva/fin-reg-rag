import numpy as np
import pytest

from finregrag.core.embeddings import LocalHashEmbeddings, OpenAIEmbeddings


@pytest.mark.parametrize(
    ("model", "expected"),
    [("text-embedding-3-small", 1536), ("text-embedding-3-large", 3072)],
)
def test_openai_embeddings_dimension_follows_known_model(model: str, expected: int) -> None:
    assert OpenAIEmbeddings(model=model, api_key="sk-test").dimension == expected


def test_openai_embeddings_probes_dimension_for_unknown_model(monkeypatch) -> None:
    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

    monkeypatch.setattr(LangchainOpenAIEmbeddings, "embed_query", lambda self, text: [0.0] * 256)

    assert OpenAIEmbeddings(model="some-future-model", api_key="sk-test").dimension == 256


def test_embed_query_is_deterministic() -> None:
    embedder = LocalHashEmbeddings(dimension=64)

    first = embedder.embed_query("nighttime transaction limit")
    second = embedder.embed_query("nighttime transaction limit")

    assert np.array_equal(first, second)


def test_embed_query_is_l2_normalized() -> None:
    embedder = LocalHashEmbeddings(dimension=64)

    vector = embedder.embed_query("some arbitrary text with several tokens")

    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_similar_texts_are_closer_than_unrelated_texts() -> None:
    embedder = LocalHashEmbeddings(dimension=256)

    a = embedder.embed_query("nighttime transaction limit for individuals")
    b = embedder.embed_query("transaction limit at night for individual users")
    c = embedder.embed_query("beneficial owners and enhanced due diligence")

    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))

    assert sim_ab > sim_ac


def test_embed_documents_empty_list_returns_empty_array() -> None:
    embedder = LocalHashEmbeddings(dimension=64)

    result = embedder.embed_documents([])

    assert result.shape == (0, 64)
