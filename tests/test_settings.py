from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import PROJECT_ROOT, Settings


def _base_env() -> dict[str, str]:
    return {"google_api_key": "test-key"}


def test_defaults_are_applied() -> None:
    settings = Settings(_env_file=None, **_base_env())

    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.max_research_steps == 5
    assert settings.max_results_per_search == 5
    assert settings.stream_final_answer is True


def test_paths_are_absolute() -> None:
    settings = Settings(_env_file=None, **_base_env())
    assert settings.chroma_path.is_absolute()
    assert settings.chroma_path == (PROJECT_ROOT / "data" / "chroma").resolve()
    assert isinstance(settings.reports_dir, Path)


def test_api_key_is_masked() -> None:
    settings = Settings(_env_file=None, **_base_env())
    assert "test-key" not in str(settings)
    assert settings.google_api_key.get_secret_value() == "test-key"


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(**_base_env(), chunk_size=500, chunk_overlap=500)


def test_log_level_is_uppercased() -> None:
    assert Settings(**_base_env(), log_level="debug").log_level == "DEBUG"
