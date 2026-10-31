from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the nighttime transaction limit for the instant payment arrangement?"])
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    total_chunks_in_store: int


class HealthResponse(BaseModel):
    status: str
    documents_in_store: int
