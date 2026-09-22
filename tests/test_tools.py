import pytest

from src.models.document import SourceDocument, SourceTool
from src.services import scraper as scraper_service
from src.services.scraper import ScrapeError
from src.tools import web_search as web_search_module
from src.tools.arxiv_tool import make_arxiv_tool
from src.tools.memory_tool import make_memory_tool
from src.tools.registry import build_tools
from src.tools.scraper_tool import make_scraper_tool
from src.tools.web_search import make_web_search_tool

HTML = """
<html><head><title>Vector Databases Explained</title></head>
<body><article><h1>Vector Databases Explained</h1>
<p>A vector database stores embeddings and supports approximate nearest neighbour
search. It is the storage backend used by retrieval augmented generation systems
to look up semantically similar passages at query time. Unlike keyword indexes it
compares dense vectors, which makes it robust against paraphrases and synonyms.
Modern implementations combine an HNSW index with metadata filtering so that
results can be narrowed down by source, language or recency before ranking.</p>
</article></body></html>
"""


def test_registry_exposes_all_five_tools(store, test_settings) -> None:
    names = {tool.name for tool in build_tools(store, test_settings)}
    assert names == {
        "search_memory",
        "web_search",
        "wikipedia_search",
        "arxiv_search",
        "scrape_webpage",
    }


def test_web_search_normalises_fields(monkeypatch, test_settings) -> None:
    def fake_search(query: str, max_results: int) -> list[dict[str, str]]:
        return [{"title": "Result A", "url": "https://a.test", "snippet": "Body A"}]

    monkeypatch.setattr(web_search_module, "run_web_search", fake_search)
    output = make_web_search_tool(test_settings).invoke({"query": "vector db"})
    assert "Result A" in output and "https://a.test" in output


def test_web_search_handles_backend_failure(monkeypatch, test_settings) -> None:
    def failing(query: str, max_results: int):
        raise RuntimeError("Web search unavailable: rate limit")

    monkeypatch.setattr(web_search_module, "run_web_search", failing)
    assert "unavailable" in make_web_search_tool(test_settings).invoke({"query": "x"})


def test_extract_document_parses_html() -> None:
    document = scraper_service.extract_document(HTML, "https://example.com/vdb", max_chars=5000)
    assert document.source_tool is SourceTool.WEB_SCRAPER
    assert "nearest neighbour" in document.text
    assert document.url == "https://example.com/vdb"


def test_extract_document_rejects_empty_page() -> None:
    with pytest.raises(ScrapeError):
        scraper_service.extract_document(
            "<html><body></body></html>", "https://x.test", max_chars=100
        )


def test_scraper_tool_stores_content(monkeypatch, store, test_settings) -> None:
    monkeypatch.setattr(scraper_service, "fetch_html", lambda url, **kwargs: HTML)
    tool = make_scraper_tool(store, test_settings)
    before = store.count()
    output = tool.invoke({"url": "https://example.com/vdb"})
    assert "Vector Databases Explained" in output
    assert store.count() > before


def test_scraper_tool_skips_fresh_url(store, test_settings) -> None:
    store.add_document(
        SourceDocument(
            source_tool=SourceTool.WEB_SCRAPER,
            url="https://cached.test",
            title="Cached",
            text="Cached content. " * 40,
        ),
        test_settings.chunk_size,
        test_settings.chunk_overlap,
    )
    output = make_scraper_tool(store, test_settings).invoke({"url": "https://cached.test"})
    assert "already fetched" in output
    assert "search_memory" in output


def test_scraper_tool_rejects_invalid_url(store, test_settings) -> None:
    assert "Invalid URL" in make_scraper_tool(store, test_settings).invoke({"url": "example.com"})


def test_memory_tool_on_empty_store(store, test_settings) -> None:
    assert "empty" in make_memory_tool(store, test_settings).invoke({"query": "anything"})


def test_memory_tool_returns_sources(store, test_settings) -> None:
    store.add_document(
        SourceDocument(
            source_tool=SourceTool.ARXIV,
            url="https://arxiv.org/abs/1234.5678",
            title="Retrieval Augmented Generation",
            text="Retrieval augmented generation for knowledge intensive tasks. " * 20,
        ),
        test_settings.chunk_size,
        test_settings.chunk_overlap,
    )
    output = make_memory_tool(store, test_settings).invoke(
        {"query": "Retrieval augmented generation for knowledge intensive tasks."}
    )
    assert "arxiv.org/abs/1234.5678" in output
    assert "relevance" in output


def test_arxiv_tool_stores_abstracts(monkeypatch, store, test_settings) -> None:
    from src.tools import arxiv_tool as arxiv_module

    def fake_search(query: str, max_results: int) -> list[SourceDocument]:
        return [
            SourceDocument(
                source_tool=SourceTool.ARXIV,
                url="https://arxiv.org/abs/2005.11401",
                title="RAG",
                text="We introduce retrieval augmented generation. " * 20,
                author="Lewis et al.",
                published_date="2020-05-22",
            )
        ]

    monkeypatch.setattr(arxiv_module, "search_papers", fake_search)
    output = make_arxiv_tool(store, test_settings).invoke({"query": "rag"})
    assert "Lewis et al." in output
    assert store.count() > 0
