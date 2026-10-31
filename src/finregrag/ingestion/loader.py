from __future__ import annotations

import os
from dataclasses import dataclass

_SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    text: str


def load_documents(docs_dir: str) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []
    if not os.path.isdir(docs_dir):
        return documents
    for filename in sorted(os.listdir(docs_dir)):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        documents.append(LoadedDocument(source=filename, text=text))
    return documents
