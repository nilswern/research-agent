"""Prüft den Report auf erfundene Quellen und kaputte Zitate."""

from __future__ import annotations

import re

from pydantic import BaseModel, computed_field

REFERENCE_HEADING = re.compile(r"^#{1,6}\s*references\b.*$", re.IGNORECASE | re.MULTILINE)
CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")
REFERENCE_ENTRY = re.compile(r"^\s*\[?(\d{1,3})[\]\.\)]", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]]+")


class ValidationResult(BaseModel):
    unsupported_urls: list[str] = []
    dangling_citations: list[int] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        return not self.unsupported_urls and not self.dangling_citations

    def as_issues(self) -> list[str]:
        issues: list[str] = []
        if self.unsupported_urls:
            issues.append(
                "URLs that were never returned by any tool: " + ", ".join(self.unsupported_urls)
            )
        if self.dangling_citations:
            issues.append(
                "Citation markers without a matching reference entry: "
                + ", ".join(f"[{n}]" for n in self.dangling_citations)
            )
        return issues


def normalize_url(url: str) -> str:
    return url.rstrip(".,;:)]").rstrip("/").lower()


def split_report(report: str) -> tuple[str, str]:
    """Trennt Fließtext und Referenzteil."""
    matches = list(REFERENCE_HEADING.finditer(report))
    if not matches:
        return report, ""
    last = matches[-1]
    return report[: last.start()], report[last.end() :]


def extract_urls(text: str) -> list[str]:
    seen: dict[str, str] = {}
    for raw in URL_PATTERN.findall(text):
        cleaned = raw.rstrip(".,;:)]")
        seen.setdefault(normalize_url(cleaned), cleaned)
    return list(seen.values())


def validate_report(report: str, allowed_urls: list[str]) -> ValidationResult:
    allowed = {normalize_url(url) for url in allowed_urls}
    unsupported = [url for url in extract_urls(report) if normalize_url(url) not in allowed]

    body, references = split_report(report)
    cited = {int(n) for n in CITATION_MARKER.findall(body)}
    defined = {int(n) for n in REFERENCE_ENTRY.findall(references)}
    dangling = sorted(cited - defined)

    return ValidationResult(unsupported_urls=unsupported, dangling_citations=dangling)
