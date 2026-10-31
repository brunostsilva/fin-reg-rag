from finregrag.core.chunking import chunk_document


def test_chunk_document_splits_long_text() -> None:
    text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
    chunks = chunk_document(text, source="doc.md", chunk_size=60, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(c.source == "doc.md" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_short_text_returns_single_chunk() -> None:
    chunks = chunk_document("short text", source="doc.md", chunk_size=800, chunk_overlap=120)

    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_chunk_document_empty_text_returns_no_chunks() -> None:
    chunks = chunk_document("", source="doc.md", chunk_size=800, chunk_overlap=120)

    assert chunks == []
