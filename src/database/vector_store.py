"""Persistent ChromaDB layer including freshness handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.logging_config import get_logger
from src.models.document import DocumentChunk, RetrievedChunk, SourceDocument
from src.services.chunking import chunk_document

logger = get_logger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        embedder: Embedder,
        cache_max_age_days: int = 30,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._cache_max_age_days = cache_max_age_days
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # embeddings are always passed explicitly
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("ChromaDB ready: %s (%d chunks)", path, self.count())

    # --- Writing ---------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        # cast: Chroma accepts nested float lists, its stubs only know the
        # ndarray/Sequence variants.
        embeddings = cast(Any, self._embedder.embed_documents([c.text for c in chunks]))
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata.to_chroma() for c in chunks],
            embeddings=embeddings,
        )
        logger.info("%d chunks stored (%s)", len(chunks), chunks[0].metadata.url)
        return len(chunks)

    def add_document(
        self,
        document: SourceDocument,
        chunk_size: int,
        chunk_overlap: int,
    ) -> int:
        self.delete_document(document.document_id)
        chunks = chunk_document(document, chunk_size, chunk_overlap)
        return self.add_chunks(chunks)

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    # --- Reading ---------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=cast(Any, [self._embedder.embed_query(text)]),
            n_results=min(top_k, self.count()),
            where=cast(Any, where),
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns these fields as None when they were not requested.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        ]

    # --- Freshness -------------------------------------------------------

    def has_fresh_document(self, url: str) -> bool:
        """True when the URL is stored and younger than CACHE_MAX_AGE_DAYS."""
        cutoff = (datetime.now(UTC) - timedelta(days=self._cache_max_age_days)).timestamp()
        found = self._collection.get(
            where=cast(
                Any,
                {
                    "$and": [
                        {"url": {"$eq": url}},
                        {"retrieved_at_ts": {"$gte": cutoff}},
                    ]
                },
            ),
            limit=1,
            include=["metadatas"],
        )
        return bool(found.get("ids"))

    def purge_stale(self) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=self._cache_max_age_days)).timestamp()
        stale = self._collection.get(
            where=cast(Any, {"retrieved_at_ts": {"$lt": cutoff}}), include=[]
        )
        stale_ids = stale.get("ids", [])
        if stale_ids:
            self._collection.delete(ids=stale_ids)
            logger.info("%d stale chunks removed", len(stale_ids))
        return len(stale_ids)

    # --- Misc ------------------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())

    def known_urls(self) -> set[str]:
        entries = self._collection.get(include=["metadatas"])
        return {str(m["url"]) for m in entries.get("metadatas", []) or [] if m and m.get("url")}
