from fastapi.testclient import TestClient

from finregrag.core.rag_pipeline import RagPipeline
from finregrag.dependencies import get_pipeline
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
