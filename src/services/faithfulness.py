"""Does the cited source actually say that?

URL validation only proves a link exists. This adds the second half: every
sentence that carries a citation marker is compared against the stored chunks
of exactly that source. The comparison reuses the embeddings that are already
in ChromaDB, so no extra model and no extra API call is involved - it costs one
vector query per cited sentence.

Similarity is a proxy, not proof: it catches a sentence that talks about
something the source never mentions, not a subtly wrong number. The threshold
is deliberately low, so that the repair stage is triggered by clear misses
rather than by paraphrase.
"""

from __future__ import annotations

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.services.validation import SupportChecker, cited_sentences, reference_urls

logger = get_logger(__name__)

SNIPPET_CHARS = 80


def _describe(sentence: str, marker: int, score: float) -> str:
    shortened = sentence if len(sentence) <= SNIPPET_CHARS else sentence[:SNIPPET_CHARS] + "..."
    return f'[{marker}] "{shortened}" (best match {score:.2f})'


def check_support(
    report: str,
    store: VectorStore,
    threshold: float,
) -> list[str]:
    """Describe every cited sentence its own source does not back up."""
    urls = reference_urls(report)
    if not urls:
        return []

    unsupported: list[str] = []
    for sentence in cited_sentences(report):
        best_score = 0.0
        best_marker = sentence.markers[0]

        for marker in sentence.markers:
            url = urls.get(marker)
            if url is None:  # dangling marker, reported by the citation check
                continue
            hits = store.query(sentence.text, top_k=1, where={"url": url})
            score = hits[0].relevance if hits else 0.0
            if score > best_score:
                best_score, best_marker = score, marker

        if best_score < threshold:
            unsupported.append(_describe(sentence.text, best_marker, best_score))

    if unsupported:
        logger.info("Faithfulness: %d claim(s) below threshold %.2f", len(unsupported), threshold)
    return unsupported


def make_support_checker(store: VectorStore, settings: Settings) -> SupportChecker | None:
    """Build the checker, or None when the feature is switched off."""
    if not settings.faithfulness_check:
        return None

    threshold = settings.faithfulness_threshold

    def checker(report: str) -> list[str]:
        return check_support(report, store, threshold)

    return checker
