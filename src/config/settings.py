"""Central configuration, loaded from .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    google_api_key: SecretStr
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # --- ChromaDB ---
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "research_documents"
    cache_max_age_days: int = Field(default=30, ge=1)

    # --- Chunking / Retrieval ---
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)

    # --- Research ---
    max_research_steps: int = Field(default=5, ge=1, le=20)
    max_results_per_search: int = Field(default=5, ge=1, le=20)
    max_report_repairs: int = Field(default=1, ge=0, le=3)

    # --- Web Scraper ---
    scraper_timeout: int = Field(default=20, ge=1, le=120)
    scraper_max_chars: int = Field(default=20_000, ge=1000)
    user_agent: str = "ResearchPilot/0.1"

    # --- Output ---
    output_language: str = "en"
    reports_dir: Path = Path("reports")
    stream_final_answer: bool = True

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("chroma_path", "reports_dir", mode="after")
    @classmethod
    def _absolutize(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level", mode="after")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def _overlap_below_size(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    def ensure_directories(self) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Required fields such as GOOGLE_API_KEY come from .env - mypy cannot see that.
    return Settings()  # type: ignore[call-arg]
