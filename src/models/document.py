"""Pydantic-Datenmodelle für Dokumente, Chunks und Retrieval-Ergebnisse."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SourceTool(StrEnum):
    GOOGLE_SEARCH = "google_search"
    WIKIPEDIA = "wikipedia_api"
    ARXIV = "arxiv_search"
    WEB_SCRAPER = "web_scraper"
    MEMORY = "search_memory"


class ContentType(StrEnum):
    WEBPAGE = "webpage"
    ENCYCLOPEDIA = "encyclopedia"
    PAPER_ABSTRACT = "paper_abstract"
    PAPER_FULLTEXT = "paper_fulltext"
    SEARCH_SNIPPET = "search_snippet"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SourceDocument(BaseModel):
    """Ein vollständiges, von einem Tool beschafftes Quelldokument."""

    model_config = ConfigDict(use_enum_values=False)

    source_tool: SourceTool
    url: str
    title: str
    text: str
    author: str | None = None
    published_date: str | None = None
    query: str | None = None
    content_type: ContentType = ContentType.WEBPAGE
    language: str = "en"
    retrieved_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def document_id(self) -> str:
        key = self.url.strip().lower() or self.content_hash
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class ChunkMetadata(BaseModel):
    """Metadaten, die pro Chunk in ChromaDB landen."""

    source_tool: str
    url: str
    title: str
    author: str | None = None
    published_date: str | None = None
    retrieved_at: str
    retrieved_at_ts: float
    query: str | None = None
    chunk_index: int
    content_hash: str
    document_id: str
    content_type: str
    language: str

    def to_chroma(self) -> dict[str, Any]:
        """Chroma erlaubt nur str/int/float/bool — None-Felder werden entfernt."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def relevance(self) -> float:
        """Cosine-Distanz -> Score in [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.distance))

    @property
    def url(self) -> str:
        return str(self.metadata.get("url", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))
