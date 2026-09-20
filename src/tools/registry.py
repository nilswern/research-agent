"""Baut die Tool-Liste für den Agenten."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.tools.arxiv_tool import make_arxiv_tool
from src.tools.memory_tool import make_memory_tool
from src.tools.scraper_tool import make_scraper_tool
from src.tools.web_search import make_web_search_tool
from src.tools.wikipedia import make_wikipedia_tool


def build_tools(store: VectorStore, settings: Settings) -> list[BaseTool]:
    return [
        make_memory_tool(store, settings),
        make_web_search_tool(settings),
        make_wikipedia_tool(store, settings),
        make_arxiv_tool(store, settings),
        make_scraper_tool(store, settings),
    ]
