from __future__ import annotations

from fastapi import APIRouter, Depends

from finregrag.api.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse, SourceChunk
from finregrag.config import Settings, get_settings
from finregrag.core.rag_pipeline import RagPipeline
from finregrag.dependencies import get_pipeline
from finregrag.ingestion.loader import load_documents

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(pipeline: RagPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(status="ok", documents_in_store=pipeline.document_count())


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RagPipeline = Depends(get_pipeline),
) -> QueryResponse:
    result = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(text=s.text, source=s.source, chunk_index=s.chunk_index, score=s.score)
            for s in result.sources
        ],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    pipeline: RagPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    documents = load_documents(settings.docs_dir)
    total_chunks = 0
    for document in documents:
        total_chunks += pipeline.ingest(
            text=document.text,
            source=document.source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    pipeline.persist()
    return IngestResponse(
        documents_loaded=len(documents),
        chunks_indexed=total_chunks,
        total_chunks_in_store=pipeline.document_count(),
    )
