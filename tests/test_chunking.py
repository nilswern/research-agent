from src.models.document import ContentType, SourceDocument, SourceTool
from src.services.chunking import chunk_document


def _document(text: str) -> SourceDocument:
    return SourceDocument(
        source_tool=SourceTool.WEB_SCRAPER,
        url="https://example.com/article",
        title="Example",
        text=text,
        content_type=ContentType.WEBPAGE,
    )


def test_chunks_carry_full_metadata() -> None:
    chunks = chunk_document(_document("Satz. " * 400), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    first = chunks[0].metadata
    assert first.source_tool == "web_scraper"
    assert first.chunk_index == 0
    assert first.document_id and first.content_hash
    assert chunks[1].metadata.chunk_index == 1


def test_chunk_ids_are_unique_and_prefixed() -> None:
    chunks = chunk_document(_document("Text. " * 400), chunk_size=300, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith(chunks[0].metadata.document_id) for i in ids)


def test_document_id_is_stable_for_same_url() -> None:
    assert _document("a").document_id == _document("b").document_id


def test_none_metadata_is_dropped_for_chroma() -> None:
    chunk = chunk_document(_document("kurz"), chunk_size=300, chunk_overlap=50)[0]
    assert "author" not in chunk.metadata.to_chroma()
    assert "url" in chunk.metadata.to_chroma()
