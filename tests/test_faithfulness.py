"""Does the cited source support the claim? Uses a bag-of-words embedder.

The 16-dimensional fake embedder from conftest collides too often to tell a
supported claim from an unrelated one, so these tests use a wider hashing
embedder where token overlap actually drives the score.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from src.database.vector_store import VectorStore
from src.models.document import SourceDocument, SourceTool
from src.services.faithfulness import check_support, make_support_checker
from src.services.validation import cited_sentences, reference_urls, validate_report

DIM = 512
URL = "https://arxiv.org/abs/2005.11401"

SUPPORTED = (
    "# RAG\n"
    "## Findings\n"
    "Retrieval augmented generation reduces hallucinations because the model is "
    "grounded in retrieved passages it can cite [1].\n"
    "## References\n"
    f"[1] RAG paper - {URL}\n"
)
UNSUPPORTED = (
    "# RAG\n"
    "## Findings\n"
    "Quantum annealing hardware factors large integers faster than classical "
    "supercomputers, which ends public key cryptography [1].\n"
    "## References\n"
    f"[1] RAG paper - {URL}\n"
)


class BagOfWordsEmbedder:
    """Hashes tokens into a wide vector, so overlap dominates the similarity."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for token in re.findall(r"\b\w+\b", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % DIM] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def faith_store(tmp_path, test_settings) -> VectorStore:
    store = VectorStore(
        path=tmp_path / "chroma",
        collection_name="faithfulness",
        embedder=BagOfWordsEmbedder(),
        cache_max_age_days=30,
    )
    store.add_document(
        SourceDocument(
            source_tool=SourceTool.ARXIV,
            url=URL,
            title="Retrieval Augmented Generation",
            text=(
                "Retrieval augmented generation grounds the model in retrieved passages "
                "and reduces hallucinations because every statement can cite the document "
                "it came from. "
            )
            * 6,
        ),
        test_settings.chunk_size,
        test_settings.chunk_overlap,
    )
    return store


def test_supported_claim_passes(faith_store, test_settings) -> None:
    assert check_support(SUPPORTED, faith_store, test_settings.faithfulness_threshold) == []


def test_unsupported_claim_is_flagged(faith_store, test_settings) -> None:
    issues = check_support(UNSUPPORTED, faith_store, test_settings.faithfulness_threshold)
    assert len(issues) == 1
    assert "Quantum annealing" in issues[0]


def test_flagged_claim_reaches_the_repair_stage(faith_store, test_settings) -> None:
    checker = make_support_checker(
        faith_store, test_settings.model_copy(update={"faithfulness_check": True})
    )
    assert checker is not None

    result = validate_report(UNSUPPORTED, [URL], support_checker=checker)
    assert not result.is_valid
    assert any("does not support" in issue for issue in result.as_issues())


def test_checker_is_off_by_default(faith_store, test_settings) -> None:
    assert test_settings.faithfulness_check is False
    assert make_support_checker(faith_store, test_settings) is None
    assert validate_report(UNSUPPORTED, [URL]).is_valid


def test_claim_without_a_reference_entry_is_left_to_the_citation_check(
    faith_store, test_settings
) -> None:
    dangling = "## Findings\nSomething entirely different [7].\n## References\n[1] RAG - " + URL
    issues = check_support(dangling, faith_store, test_settings.faithfulness_threshold)
    assert issues and "[7]" in issues[0]


def test_reference_urls_and_cited_sentences() -> None:
    assert reference_urls(SUPPORTED) == {1: URL}

    sentences = cited_sentences(SUPPORTED)
    assert len(sentences) == 1
    assert sentences[0].markers == [1]
    assert "hallucinations" in sentences[0].text
