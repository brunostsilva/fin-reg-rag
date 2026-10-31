import numpy as np

from finregrag.core.embeddings import LocalHashEmbeddings


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
