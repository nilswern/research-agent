"""Gemeinsame Helfer für die Tool-Ausgabe."""

from __future__ import annotations

MAX_SNIPPET_CHARS = 400


def truncate(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "..."


def no_results(query: str, source: str) -> str:
    return f"No results from {source} for '{query}'. Try a different or broader query."
