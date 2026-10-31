import os

from fastapi.testclient import TestClient

from finregrag.config import Settings, get_settings
from finregrag.core.rag_pipeline import RagPipeline
from finregrag.dependencies import build_pipeline, get_pipeline
from finregrag.main import app


def make_client(pipeline: RagPipeline) -> TestClient:
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def test_health_reports_document_count(ingested_pipeline: RagPipeline) -> None:
    client = make_client(ingested_pipeline)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents_in_store"] == ingested_pipeline.document_count()


def test_query_endpoint_returns_answer_and_sources(ingested_pipeline: RagPipeline) -> None:
    client = make_client(ingested_pipeline)

    response = client.post("/query", json={"question": "What is the nighttime transaction limit?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"] == "pix.md"


def test_query_endpoint_validates_empty_question(ingested_pipeline: RagPipeline) -> None:
    client = make_client(ingested_pipeline)

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_ingest_endpoint_persists_faiss_index(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "pix.md").write_text("# Limits\n\nNighttime limit is R$ 1,000.00.", encoding="utf-8")
    index_path = str(tmp_path / "index" / "faiss_index")
    settings = Settings(docs_dir=str(docs_dir), faiss_index_path=index_path)
    app.dependency_overrides[get_settings] = lambda: settings
    client = make_client(build_pipeline(settings))

    response = client.post("/ingest")

    assert response.status_code == 200
    assert os.path.exists(index_path)
    assert build_pipeline(settings).document_count() == response.json()["total_chunks_in_store"]
