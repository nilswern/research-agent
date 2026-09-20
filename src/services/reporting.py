"""Saves reports as Markdown files."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from src.config.logging_config import get_logger

logger = get_logger(__name__)

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 60) -> str:
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = _NON_SLUG.sub("-", ascii_text).strip("-")
    return slug[:max_length].rstrip("-") or "report"


def build_frontmatter(
    question: str,
    model: str,
    research_steps: int,
    sources: list[str],
    created_at: datetime,
) -> str:
    lines = [
        "---",
        f'question: "{question.replace(chr(34), chr(39))}"',
        f"model: {model}",
        f"research_steps: {research_steps}",
        f"created_at: {created_at.isoformat()}",
        "sources:",
        *[f"  - {url}" for url in sources],
        "---",
        "",
    ]
    return "\n".join(lines)


def save_report(
    report: str,
    question: str,
    reports_dir: Path,
    model: str,
    research_steps: int,
    sources: list[str],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC)
    filename = f"{created_at.strftime('%Y%m%d-%H%M%S')}_{slugify(question)}.md"
    path = reports_dir / filename
    path.write_text(
        build_frontmatter(question, model, research_steps, sources, created_at) + report,
        encoding="utf-8",
    )
    logger.info("Report saved: %s", path)
    return path
