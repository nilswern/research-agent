"""Factory for the VectorStore and its local embedder."""

from __future__ import annotations

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.services.embeddings import EmbeddingService


def create_store(settings: Settings) -> VectorStore:
    embedder = EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    return VectorStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        embedder=embedder,
        cache_max_age_days=settings.cache_max_age_days,
    )
