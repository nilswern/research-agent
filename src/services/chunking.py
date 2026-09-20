"""Zerlegt Quelldokumente in Chunks inkl. Metadaten."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.document import ChunkMetadata, DocumentChunk, SourceDocument


def chunk_document(
    document: SourceDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    pieces = [p.strip() for p in splitter.split_text(document.text) if p.strip()]

    chunks: list[DocumentChunk] = []
    for index, piece in enumerate(pieces):
        metadata = ChunkMetadata(
            source_tool=document.source_tool.value,
            url=document.url,
            title=document.title,
            author=document.author,
            published_date=document.published_date,
            retrieved_at=document.retrieved_at.isoformat(),
            retrieved_at_ts=document.retrieved_at.timestamp(),
            query=document.query,
            chunk_index=index,
            content_hash=document.content_hash,
            document_id=document.document_id,
            content_type=document.content_type.value,
            language=document.language,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:{index}",
                text=piece,
                metadata=metadata,
            )
        )
    return chunks
