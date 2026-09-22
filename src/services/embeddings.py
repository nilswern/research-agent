"""Local embedding service (sentence-transformers, no API quota)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.logging_config import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps model loading and the E5 query:/passage: prefixes."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (%s)", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def _needs_e5_prefix(self) -> bool:
        return "e5" in self.model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"passage: {t}" for t in texts] if self._needs_e5_prefix else texts
        return self._encode(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = f"query: {text}" if self._needs_e5_prefix else text
        return self._encode([prepared])[0]
