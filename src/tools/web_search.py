"""google_search - general web search via DuckDuckGo (no API key)."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.tools._common import no_results, truncate

logger = get_logger(__name__)


def run_web_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Raw search; normalises the field names across ddgs versions."""
    from ddgs import DDGS

    try:
        with DDGS() as client:
            raw: list[dict[str, Any]] = list(client.text(query=query, max_results=max_results))
    except Exception as exc:
        logger.warning("Web search failed for '%s': %s", query, exc)
        raise RuntimeError(f"Web search unavailable: {exc}") from exc

    results: list[dict[str, str]] = []
    for item in raw:
        url = item.get("href") or item.get("url") or item.get("link") or ""
        if not url:
            continue
        results.append(
            {
                "title": item.get("title") or url,
                "url": url,
                "snippet": item.get("body") or item.get("snippet") or "",
            }
        )
    return results


def make_web_search_tool(settings: Settings) -> BaseTool:
    @tool("google_search")
    def google_search(query: str) -> str:
        """Search the public web for a topic and return ranked titles, URLs and short snippets.

        Use this to discover sources for general, current or non-academic questions.
        Snippets are short on purpose: pick the most promising URLs and call
        scrape_webpage on them to read the full content.
        """
        try:
            results = run_web_search(query, settings.max_results_per_search)
        except RuntimeError as exc:
            return f"Web search unavailable: {exc}"
        if not results:
            return no_results(query, "web search")

        lines = [f"Web search results for '{query}':"]
        for index, result in enumerate(results, start=1):
            lines.append(
                f"{index}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   Snippet: {truncate(result['snippet'])}"
            )
        return "\n".join(lines)

    return google_search
