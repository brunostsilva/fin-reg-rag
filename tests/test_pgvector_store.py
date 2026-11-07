"""PgVectorStore against a real Postgres with the pgvector extension, started by Testcontainers.

Skipped when Docker isn't available; GitHub Actions runners have it.
"""

import numpy as np
import psycopg
import pytest

from finregrag.core.vectorstore.pgvector_store import PgVectorStore

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"  # the image used by docker-compose.yml
DIMENSION = 8


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="needs Docker for the Postgres + pgvector container")


@pytest.fixture(scope="module")
def postgres_dsn():
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(PGVECTOR_IMAGE, driver=None) as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def store(postgres_dsn: str) -> PgVectorStore:
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS document_chunks")
    return PgVectorStore(dsn=postgres_dsn, dimension=DIMENSION)


def _unit(*values: float) -> np.ndarray:
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def test_search_returns_the_closest_chunks_first_with_cosine_scores(store: PgVectorStore) -> None:
    store.add(
        np.stack([_unit(1, 0), _unit(0, 1), _unit(1, 1)]),
        texts=["nighttime limit", "record retention", "both topics"],
        sources=["pix.md", "aml.md", "mixed.md"],
        chunk_indices=[0, 4, 2],
    )

    results = store.search(_unit(1, 0), top_k=2)

    assert [r.text for r in results] == ["nighttime limit", "both topics"]
    assert (results[0].source, results[0].chunk_index) == ("pix.md", 0)
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(np.cos(np.pi / 4), abs=1e-4)


def test_delete_source_removes_only_its_chunks(store: PgVectorStore) -> None:
    store.add(
        np.stack([_unit(1, 0), _unit(1, 1), _unit(0, 1)]),
        texts=["a", "b", "c"],
        sources=["pix.md", "pix.md", "aml.md"],
        chunk_indices=[0, 1, 0],
    )

    assert store.count() == 3
    assert store.delete_source("pix.md") == 2
    assert store.delete_source("unknown.md") == 0
    assert store.count() == 1
    assert [r.source for r in store.search(_unit(1, 0), top_k=5)] == ["aml.md"]


def test_adding_no_chunks_is_a_noop(store: PgVectorStore) -> None:
    store.add(np.empty((0, DIMENSION), dtype=np.float32), texts=[], sources=[], chunk_indices=[])

    assert store.count() == 0
    assert store.search(_unit(1, 0), top_k=3) == []


def test_chunks_are_visible_from_a_new_store_on_the_same_database(store: PgVectorStore, postgres_dsn: str) -> None:
    store.add(np.stack([_unit(1, 0)]), texts=["persisted"], sources=["pix.md"], chunk_indices=[0])

    reopened = PgVectorStore(dsn=postgres_dsn, dimension=DIMENSION)

    assert reopened.count() == 1
    assert reopened.search(_unit(1, 0), top_k=1)[0].text == "persisted"


def test_reopening_with_a_different_dimension_fails_fast(store: PgVectorStore, postgres_dsn: str) -> None:
    with pytest.raises(ValueError, match=f"dimension {DIMENSION}"):
        PgVectorStore(dsn=postgres_dsn, dimension=DIMENSION * 2)
