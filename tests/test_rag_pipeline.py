from finregrag.core.rag_pipeline import RagPipeline


def test_ingest_returns_number_of_chunks_created(pipeline: RagPipeline) -> None:
    chunks_created = pipeline.ingest(
        text="# Title\n\nSome content about transaction limits.",
        source="doc.md",
        chunk_size=200,
        chunk_overlap=20,
    )

    assert chunks_created >= 1
    assert pipeline.document_count() == chunks_created


def test_query_returns_relevant_source(ingested_pipeline: RagPipeline) -> None:
    result = ingested_pipeline.query("What is the nighttime transaction limit?")

    assert result.sources
    assert result.sources[0].source == "pix.md"
    assert "1,000.00" in result.answer


def test_query_on_empty_store_returns_no_sources(pipeline: RagPipeline) -> None:
    result = pipeline.query("anything")

    assert result.sources == []
    assert "couldn't find" in result.answer.lower()


def test_query_respects_top_k_override(ingested_pipeline: RagPipeline) -> None:
    result = ingested_pipeline.query("record retention and transaction limits", top_k=1)

    assert len(result.sources) == 1
