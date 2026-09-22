"""scrape_webpage - full text of one URL, with a freshness check."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.services.scraper import ScrapeError, scrape_url
from src.services.untrusted import sanitize_line
from src.tools._common import fetched

logger = get_logger(__name__)


def make_scraper_tool(store: VectorStore, settings: Settings) -> BaseTool:
    @tool("scrape_webpage")
    def scrape_webpage(url: str) -> str:
        """Fetch a single web page and extract its main text content.

        Call this after a search when a result looks promising and the snippet is not
        enough. Pass one full URL (starting with http:// or https://). The extracted
        text is stored in memory, so a page already fetched recently is not re-fetched.
        """
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return f"Invalid URL '{url}'. Provide a full URL starting with http:// or https://."

        if store.has_fresh_document(url):
            return (
                f"{url} was already fetched recently and its content is in memory. "
                f"Use search_memory with a specific question to retrieve it."
            )

        try:
            document = scrape_url(
                url,
                timeout=settings.scraper_timeout,
                max_chars=settings.scraper_max_chars,
                user_agent=settings.user_agent,
            )
        except ScrapeError as exc:
            logger.warning("Scrape failed: %s", exc)
            return f"Could not read {url}: {exc}"

        store.add_document(document, settings.chunk_size, settings.chunk_overlap)
        return (
            f"Scraped: {sanitize_line(document.title)}\n"
            f"URL: {document.url}\n"
            f"Published: {document.published_date or 'unknown'}\n\n"
            f"{fetched(document.text, 3000)}"
        )

    return scrape_webpage
