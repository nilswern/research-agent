"""Pydantic data models for documents, chunks and retrieval results."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SourceTool(StrEnum):
    """The tool that fetched a document; the value is the tool's own name."""

    WEB_SEARCH = "web_search"
    WIKIPEDIA = "wikipedia_search"
    ARXIV = "arxiv_search"
    WEB_SCRAPER = "scrape_webpage"


class ContentType(StrEnum):
    WEBPAGE = "webpage"
    ENCYCLOPEDIA = "encyclopedia"
    PAPER_ABSTRACT = "paper_abstract"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SourceDocument(BaseModel):
    """A complete source document fetched by one of the tools."""

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
    """Metadata stored in ChromaDB for every chunk."""

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
        """Chroma only allows str/int/float/bool - None fields are dropped."""
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
        """Cosine distance -> score in [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.distance))

    @property
    def url(self) -> str:
        return str(self.metadata.get("url", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))
