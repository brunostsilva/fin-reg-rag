from __future__ import annotations

import pytest

from finregrag.core.embeddings import LocalHashEmbeddings
from finregrag.core.llm.fake_llm import FakeExtractiveLLM
from finregrag.core.rag_pipeline import RagPipeline
from finregrag.core.vectorstore.faiss_store import FaissVectorStore

SAMPLE_DOCS = {
    "pix.md": (
        "# Instant Payment Rules\n\n"
        "## Transaction Limits\n"
        "The nighttime transaction limit for individuals is R$ 1,000.00 per transaction, "
        "applicable between 8 PM and 6 AM.\n"
    ),
    "aml.md": (
        "# AML Requirements\n\n"
        "## Record Retention\n"
        "Customer identification and transaction records must be kept for at least 5 years "
        "after the end of the business relationship.\n"
    ),
}


@pytest.fixture
def pipeline(tmp_path) -> RagPipeline:
    embeddings = LocalHashEmbeddings(dimension=128)
    vector_store = FaissVectorStore(dimension=embeddings.dimension, index_path=str(tmp_path / "index.faiss"))
    llm = FakeExtractiveLLM()
    return RagPipeline(embeddings=embeddings, vector_store=vector_store, llm=llm, top_k=2)


@pytest.fixture
def ingested_pipeline(pipeline: RagPipeline) -> RagPipeline:
    for source, text in SAMPLE_DOCS.items():
        pipeline.ingest(text=text, source=source, chunk_size=200, chunk_overlap=20)
    return pipeline


@pytest.fixture(autouse=True)
def _clear_fastapi_dependency_overrides():
    yield
    from finregrag.main import app

    app.dependency_overrides.clear()
