import numpy as np
import pytest

from finregrag.core.vectorstore.faiss_store import FaissVectorStore


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def test_add_and_search_returns_closest_match(tmp_path) -> None:
    store = FaissVectorStore(dimension=4, index_path=str(tmp_path / "index.faiss"))
    embeddings = np.array(
        [
            _unit(np.array([1.0, 0.0, 0.0, 0.0])),
            _unit(np.array([0.0, 1.0, 0.0, 0.0])),
        ],
        dtype=np.float32,
    )

    store.add(embeddings, texts=["doc a", "doc b"], sources=["a.md", "b.md"], chunk_indices=[0, 0])
    results = store.search(_unit(np.array([0.9, 0.1, 0.0, 0.0])), top_k=1)

    assert len(results) == 1
    assert results[0].text == "doc a"
    assert results[0].source == "a.md"


def test_search_on_empty_store_returns_no_results(tmp_path) -> None:
    store = FaissVectorStore(dimension=4, index_path=str(tmp_path / "index.faiss"))

    results = store.search(np.zeros(4, dtype=np.float32), top_k=3)

    assert results == []


def test_persist_and_reload_preserves_data(tmp_path) -> None:
    index_path = str(tmp_path / "index.faiss")
    store = FaissVectorStore(dimension=4, index_path=index_path)
    store.add(
        np.array([_unit(np.array([1.0, 0.0, 0.0, 0.0]))], dtype=np.float32),
        texts=["persisted doc"],
        sources=["p.md"],
        chunk_indices=[0],
    )
    store.persist()

    reloaded = FaissVectorStore(dimension=4, index_path=index_path)

    assert reloaded.count() == 1
    results = reloaded.search(_unit(np.array([1.0, 0.0, 0.0, 0.0])), top_k=1)
    assert results[0].text == "persisted doc"


def test_delete_source_removes_only_its_chunks_and_keeps_metadata_aligned(tmp_path) -> None:
    store = FaissVectorStore(dimension=4, index_path=str(tmp_path / "index.faiss"))
    store.add(
        np.eye(4, dtype=np.float32),
        texts=["a0", "b0", "a1", "c0"],
        sources=["a.md", "b.md", "a.md", "c.md"],
        chunk_indices=[0, 0, 1, 0],
    )

    removed = store.delete_source("a.md")

    assert removed == 2
    assert store.count() == 2
    assert store.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), top_k=1)[0].text == "b0"
    assert store.search(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), top_k=1)[0].text == "c0"


def test_delete_unknown_source_is_a_noop(tmp_path) -> None:
    store = FaissVectorStore(dimension=4, index_path=str(tmp_path / "index.faiss"))
    store.add(np.eye(1, 4, dtype=np.float32), texts=["a"], sources=["a.md"], chunk_indices=[0])

    assert store.delete_source("missing.md") == 0
    assert store.count() == 1


@pytest.mark.parametrize("top_k", [1, 2, 5])
def test_search_never_returns_more_than_top_k(tmp_path, top_k: int) -> None:
    store = FaissVectorStore(dimension=4, index_path=str(tmp_path / "index.faiss"))
    embeddings = np.eye(4, dtype=np.float32)
    store.add(embeddings, texts=["a", "b", "c", "d"], sources=["s"] * 4, chunk_indices=[0, 1, 2, 3])

    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=top_k)

    assert len(results) <= min(top_k, 4)
