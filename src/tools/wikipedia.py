"""wikipedia_search - encyclopedic overviews, full text goes into ChromaDB."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from langchain_core.tools import BaseTool, tool

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.models.document import ContentType, SourceDocument, SourceTool
from src.services.untrusted import sanitize_line
from src.tools._common import fetched, no_results

logger = get_logger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
MAX_ARTICLE_CHARS = 12_000


def search_titles(query: str, limit: int, *, timeout: int, user_agent: str) -> list[str]:
    response = httpx.get(
        API_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    return [hit["title"] for hit in hits if hit.get("title")]


def fetch_article(
    title: str, *, user_agent: str, query: str | None = None
) -> SourceDocument | None:
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    page = wiki.page(title)
    if not page.exists():
        return None
    return SourceDocument(
        source_tool=SourceTool.WIKIPEDIA,
        url=page.fullurl,
        title=page.title,
        text=page.text[:MAX_ARTICLE_CHARS],
        query=query,
        content_type=ContentType.ENCYCLOPEDIA,
        language="en",
        retrieved_at=datetime.now(UTC),
    )


def make_wikipedia_tool(store: VectorStore, settings: Settings) -> BaseTool:
    @tool("wikipedia_search")
    def wikipedia_search(query: str) -> str:
        """Look up encyclopedic background knowledge on Wikipedia.

        Best for definitions, established facts, historical context and overviews of
        well-known entities. Returns article summaries; the full articles are stored
        in memory and can be re-read later with search_memory.
        """
        try:
            titles = search_titles(
                query,
                limit=min(settings.max_results_per_search, 3),
                timeout=settings.scraper_timeout,
                user_agent=settings.user_agent,
            )
        except httpx.HTTPError as exc:
            logger.warning("Wikipedia search failed for '%s': %s", query, exc)
            return f"Wikipedia unavailable: {exc}"

        if not titles:
            return no_results(query, "Wikipedia")

        blocks: list[str] = []
        for title in titles:
            try:
                document = fetch_article(title, user_agent=settings.user_agent, query=query)
            except Exception as exc:
                logger.warning("Wikipedia article '%s' failed: %s", title, exc)
                continue
            if document is None:
                continue
            store.add_document(document, settings.chunk_size, settings.chunk_overlap)
            blocks.append(
                f"Title: {sanitize_line(document.title)}\n"
                f"URL: {document.url}\n"
                f"Summary: {fetched(document.text, 800)}"
            )

        if not blocks:
            return no_results(query, "Wikipedia")
        return f"Wikipedia results for '{query}':\n\n" + "\n\n".join(blocks)

    return wikipedia_search
