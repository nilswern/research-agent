"""Persistenter ChromaDB-Layer inkl. Freshness-Handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.models.document import DocumentChunk, RetrievedChunk, SourceDocument
from src.services.chunking import chunk_document
from src.services.embeddings import get_embedding_service

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
            embedding_function=None,  # Embeddings werden immer explizit übergeben
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("ChromaDB bereit: %s (%d Chunks)", path, self.count())

    # --- Schreiben -------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        # cast: Chroma nimmt verschachtelte Float-Listen an, die Stubs kennen
        # aber nur ndarray-/Sequence-Varianten.
        embeddings = cast(Any, self._embedder.embed_documents([c.text for c in chunks]))
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata.to_chroma() for c in chunks],
            embeddings=embeddings,
        )
        logger.info("%d Chunks gespeichert (%s)", len(chunks), chunks[0].metadata.url)
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

    # --- Lesen -----------------------------------------------------------

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
        # Chroma liefert die Felder als None, wenn sie nicht angefordert wurden.
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
        """True, wenn die URL gespeichert und jünger als CACHE_MAX_AGE_DAYS ist."""
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
            logger.info("%d veraltete Chunks entfernt", len(stale_ids))
        return len(stale_ids)

    # --- Sonstiges -------------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())

    def known_urls(self) -> set[str]:
        entries = self._collection.get(include=["metadatas"])
        return {str(m["url"]) for m in entries.get("metadatas", []) or [] if m and m.get("url")}


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        embedder=get_embedding_service(),
        cache_max_age_days=settings.cache_max_age_days,
    )
