"""search_memory - semantic search over material collected earlier."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.tools._common import fetched, label


def make_memory_tool(store: VectorStore, settings: Settings) -> BaseTool:
    @tool("search_memory")
    def search_memory(query: str) -> str:
        """Search research material collected earlier (this run or previous runs).

        Always try this first before running a new web search: the answer may already
        be stored. Returns the most relevant passages with their source URL and a
        relevance score between 0 and 1.
        """
        if store.count() == 0:
            return "Memory is empty. Use google_search, wikipedia_search or arxiv_search first."

        hits = store.query(query, top_k=settings.retrieval_top_k)
        if not hits:
            return f"Nothing relevant in memory for '{query}'."

        blocks = [
            f"[{index}] {label(hit.title)} (relevance {hit.relevance:.2f})\n"
            f"    URL: {hit.url}\n"
            f"    Source: {hit.metadata.get('source_tool', 'unknown')}"
            f" | retrieved {str(hit.metadata.get('retrieved_at', ''))[:10]}\n"
            f"    {fetched(hit.text, 700)}"
            for index, hit in enumerate(hits, start=1)
        ]
        return f"Memory results for '{query}':\n\n" + "\n\n".join(blocks)

    return search_memory
