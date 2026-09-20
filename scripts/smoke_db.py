"""Smoke test with the real embedding model and a real ChromaDB."""

from __future__ import annotations

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.database.vector_store import get_vector_store
from src.models.document import ContentType, SourceDocument, SourceTool


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()

    store = get_vector_store()
    print(f"Chunks before the run: {store.count()}")

    document = SourceDocument(
        source_tool=SourceTool.WIKIPEDIA,
        url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        title="Retrieval-augmented generation",
        text=(
            "Retrieval-augmented generation combines a retrieval system with a "
            "language model. Relevant documents are fetched from a knowledge base "
            "and passed to the model as additional context. This reduces "
            "hallucinations and allows citing sources. "
        ) * 12,
        content_type=ContentType.ENCYCLOPEDIA,
        query="what is retrieval augmented generation",
        language="en",
    )
    added = store.add_document(document, settings.chunk_size, settings.chunk_overlap)
    print(f"Newly stored chunks: {added}")
    print(f"Fresh? {store.has_fresh_document(document.url)}")

    for hit in store.query("how does RAG reduce hallucinations", top_k=3):
        print(f"[{hit.relevance:.3f}] {hit.title} -> {hit.text[:90]}...")


if __name__ == "__main__":
    main()