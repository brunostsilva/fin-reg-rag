"""Exercises the app through an async client (pytest-asyncio + httpx ASGITransport).

The sync TestClient tests in test_api.py run the app on a separate event loop; these run it on the
test's own loop, the way an async caller (e.g. another ASGI service) would.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio

from finregrag.core.rag_pipeline import RagPipeline
from finregrag.dependencies import get_pipeline
from finregrag.main import app

QUESTION = "What is the nighttime transaction limit?"


@pytest_asyncio.fixture
async def client(ingested_pipeline: RagPipeline):
    app.dependency_overrides[get_pipeline] = lambda: ingested_pipeline
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_health_over_async_client(client: httpx.AsyncClient, ingested_pipeline: RagPipeline) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents_in_store": ingested_pipeline.document_count()}


@pytest.mark.asyncio
async def test_query_over_async_client(client: httpx.AsyncClient) -> None:
    response = await client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "pix.md"


@pytest.mark.asyncio
async def test_concurrent_queries_return_the_same_answer(client: httpx.AsyncClient) -> None:
    responses = await asyncio.gather(*(client.post("/query", json={"question": QUESTION}) for _ in range(10)))

    assert {r.status_code for r in responses} == {200}
    assert len({r.json()["answer"] for r in responses}) == 1
