"""Shared helpers for tool output."""

from __future__ import annotations

from src.services.untrusted import sanitize_line, wrap_untrusted

MAX_SNIPPET_CHARS = 400


def truncate(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "..."


def fetched(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    """Third-party text: truncated and fenced as data the model must not obey."""
    return wrap_untrusted(truncate(text, limit))


def label(text: str) -> str:
    """Third-party metadata (titles, authors) as one harmless line."""
    return sanitize_line(text)


def no_results(query: str, source: str) -> str:
    return f"No results from {source} for '{query}'. Try a different or broader query."
