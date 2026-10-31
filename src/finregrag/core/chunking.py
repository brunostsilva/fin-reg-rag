from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_document(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    pieces = splitter.split_text(text)
    return [
        Chunk(text=piece, source=source, chunk_index=i)
        for i, piece in enumerate(pieces)
    ]
