from __future__ import annotations

from fastapi import FastAPI

from finregrag.api.routes import router

app = FastAPI(
    title="FinRegRAG",
    description="RAG service for querying financial/payments regulation documents.",
    version="0.1.0",
)
app.include_router(router)
