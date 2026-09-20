"""Smoke test of the real tools (real network calls)."""

from __future__ import annotations

import sys

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.database.vector_store import get_vector_store
from src.tools import build_tools


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()

    tools = {tool.name: tool for tool in build_tools(get_vector_store(), settings)}
    query = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"

    for name in ("google_search", "wikipedia_search", "arxiv_search"):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(tools[name].invoke({"query": query})[:1200])

    print(f"\n{'=' * 70}\nsearch_memory\n{'=' * 70}")
    print(tools["search_memory"].invoke({"query": query})[:1200])


if __name__ == "__main__":
    main()