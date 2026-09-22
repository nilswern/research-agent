"""Checks the report for invented sources and broken citations."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, computed_field

REFERENCE_HEADING = re.compile(r"^#{1,6}\s*references\b.*$", re.IGNORECASE | re.MULTILINE)
CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")
REFERENCE_ENTRY = re.compile(r"^\s*\[?(\d{1,3})[\]\.\)]", re.MULTILINE)
REFERENCE_LINE = re.compile(r"^\s*\[?(\d{1,3})[\]\.\)]\s*(.*)$", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

#: Given the full report, return a description for every claim that its cited
#: source does not back up. Implemented in :mod:`src.services.faithfulness`.
SupportChecker = Callable[[str], list[str]]


class CitedSentence(BaseModel):
    text: str
    markers: list[int]


class ValidationResult(BaseModel):
    unsupported_urls: list[str] = []
    dangling_citations: list[int] = []
    unsupported_claims: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        return not (self.unsupported_urls or self.dangling_citations or self.unsupported_claims)

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
        if self.unsupported_claims:
            issues.append(
                "Claims their cited source does not support: " + "; ".join(self.unsupported_claims)
            )
        return issues


def normalize_url(url: str) -> str:
    return url.rstrip(".,;:)]").rstrip("/").lower()


def find_unsupported_urls(text: str, allowed_urls: list[str]) -> list[str]:
    """URLs in ``text`` that no tool ever returned."""
    allowed = {normalize_url(url) for url in allowed_urls}
    return [url for url in extract_urls(text) if normalize_url(url) not in allowed]


def split_report(report: str) -> tuple[str, str]:
    """Split the body text from the reference section."""
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


def reference_urls(report: str) -> dict[int, str]:
    """Map every reference marker to the URL its entry points at."""
    _, references = split_report(report)
    mapping: dict[int, str] = {}
    for marker, entry in REFERENCE_LINE.findall(references):
        urls = extract_urls(entry)
        if urls:
            mapping[int(marker)] = urls[0]
    return mapping


def cited_sentences(report: str) -> list[CitedSentence]:
    """Sentences of the body that carry at least one citation marker."""
    body, _ = split_report(report)
    sentences: list[CitedSentence] = []
    for raw in SENTENCE_SPLIT.split(body):
        sentence = raw.strip()
        if not sentence or sentence.startswith("#"):
            continue
        markers = sorted({int(n) for n in CITATION_MARKER.findall(sentence)})
        if markers:
            sentences.append(CitedSentence(text=sentence, markers=markers))
    return sentences


def validate_report(
    report: str,
    allowed_urls: list[str],
    support_checker: SupportChecker | None = None,
) -> ValidationResult:
    body, references = split_report(report)
    cited = {int(n) for n in CITATION_MARKER.findall(body)}
    defined = {int(n) for n in REFERENCE_ENTRY.findall(references)}

    return ValidationResult(
        unsupported_urls=find_unsupported_urls(report, allowed_urls),
        dangling_citations=sorted(cited - defined),
        unsupported_claims=support_checker(report) if support_checker else [],
    )
