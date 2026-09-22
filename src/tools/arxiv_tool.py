"""arxiv_search - scientific papers, abstracts only (no PDF full text)."""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.tools import BaseTool, tool

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.models.document import ContentType, SourceDocument, SourceTool
from src.tools._common import fetched, label, no_results

logger = get_logger(__name__)


def search_papers(query: str, max_results: int) -> list[SourceDocument]:
    import arxiv

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    documents: list[SourceDocument] = []
    for result in client.results(search):
        authors = ", ".join(author.name for author in result.authors[:6])
        documents.append(
            SourceDocument(
                source_tool=SourceTool.ARXIV,
                url=result.entry_id,
                title=result.title.strip(),
                text=result.summary.strip(),
                author=authors or None,
                published_date=result.published.date().isoformat() if result.published else None,
                query=query,
                content_type=ContentType.PAPER_ABSTRACT,
                language="en",
                retrieved_at=datetime.now(UTC),
            )
        )
    return documents


def make_arxiv_tool(store: VectorStore, settings: Settings) -> BaseTool:
    @tool("arxiv_search")
    def arxiv_search(query: str) -> str:
        """Search arXiv for scientific papers and return title, authors, date and abstract.

        Use this for research questions, technical methods, benchmarks or any claim
        that should be backed by peer-reviewed or preprint literature. Only abstracts
        are returned; if an abstract is not enough, call scrape_webpage on the paper URL.
        """
        try:
            documents = search_papers(query, settings.max_results_per_search)
        except Exception as exc:
            logger.warning("arXiv search failed for '%s': %s", query, exc)
            return f"arXiv unavailable: {exc}"

        if not documents:
            return no_results(query, "arXiv")

        blocks: list[str] = []
        for document in documents:
            store.add_document(document, settings.chunk_size, settings.chunk_overlap)
            blocks.append(
                f"Title: {label(document.title)}\n"
                f"Authors: {label(document.author or 'unknown')}\n"
                f"Published: {document.published_date or 'unknown'}\n"
                f"URL: {document.url}\n"
                f"Abstract: {fetched(document.text, 900)}"
            )
        return f"arXiv results for '{query}':\n\n" + "\n\n".join(blocks)

    return arxiv_search
