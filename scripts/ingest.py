#!/usr/bin/env python3
"""CLI helper to (re)index the documents in DOCS_DIR without starting the API.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finregrag.config import get_settings
from finregrag.dependencies import build_pipeline
from finregrag.ingestion.loader import load_documents


def main() -> None:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    documents = load_documents(settings.docs_dir)
    if not documents:
        print(f"No documents found in {settings.docs_dir!r}.")
        return

    total_chunks = 0
    for document in documents:
        chunks = pipeline.ingest(
            text=document.text,
            source=document.source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        total_chunks += chunks
        print(f"  - {document.source}: {chunks} chunks")

    pipeline.persist()

    print(f"\nTotal: {len(documents)} documents, {total_chunks} chunks indexed.")
    print(f"Chunks in vector store: {pipeline.document_count()}")


if __name__ == "__main__":
    main()
