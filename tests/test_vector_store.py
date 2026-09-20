from datetime import UTC, datetime, timedelta

from src.models.document import SourceDocument, SourceTool


def _document(url: str, text: str, retrieved_at: datetime | None = None) -> SourceDocument:
    return SourceDocument(
        source_tool=SourceTool.WIKIPEDIA,
        url=url,
        title=f"Doc {url}",
        text=text,
        retrieved_at=retrieved_at or datetime.now(UTC),
    )


def test_add_and_query(store) -> None:
    store.add_document(_document("https://a.test", "Quantum computing basics. " * 30), 300, 50)
    store.add_document(_document("https://b.test", "Cooking pasta at home. " * 30), 300, 50)

    results = store.query("Quantum computing basics.", top_k=3)
    assert results
    assert results[0].url == "https://a.test"
    assert 0.0 <= results[0].relevance <= 1.0


def test_query_on_empty_store_returns_empty_list(store) -> None:
    assert store.query("irgendwas") == []


def test_reindexing_same_url_does_not_duplicate(store) -> None:
    document = _document("https://a.test", "Stabiler Inhalt. " * 30)
    store.add_document(document, 300, 50)
    count_after_first = store.count()
    store.add_document(document, 300, 50)
    assert store.count() == count_after_first


def test_freshness_check(store) -> None:
    store.add_document(_document("https://fresh.test", "Neu. " * 30), 300, 50)
    old = datetime.now(UTC) - timedelta(days=90)
    store.add_document(_document("https://old.test", "Alt. " * 30, old), 300, 50)

    assert store.has_fresh_document("https://fresh.test") is True
    assert store.has_fresh_document("https://old.test") is False
    assert store.has_fresh_document("https://unknown.test") is False


def test_purge_stale_removes_only_old_chunks(store) -> None:
    store.add_document(_document("https://fresh.test", "Neu. " * 30), 300, 50)
    old = datetime.now(UTC) - timedelta(days=90)
    store.add_document(_document("https://old.test", "Alt. " * 30, old), 300, 50)

    removed = store.purge_stale()
    assert removed > 0
    assert store.known_urls() == {"https://fresh.test"}


def test_persistence_across_instances(store, tmp_path) -> None:
    from src.database.vector_store import VectorStore
    from tests.conftest import FakeEmbedder

    store.add_document(_document("https://a.test", "Persistenz. " * 30), 300, 50)
    reopened = VectorStore(tmp_path / "chroma", "test_collection", FakeEmbedder(), 30)
    assert reopened.count() == store.count()
