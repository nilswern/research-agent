"""The eval harness, exercised offline: metrics and rendering, never the API."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from evals.run_eval import (
    EvalQuestion,
    QuestionResult,
    aggregate,
    load_questions,
    run_question,
    to_markdown,
)
from tests.conftest import FakeChatModel, seed_rag_document, tool_call_message

SOURCE_URL = "https://arxiv.org/abs/2005.11401"
GOOD_REPORT = (
    "# RAG\n## Summary\nIt combines retrieval and generation [1].\n"
    f"## References\n[1] RAG paper - {SOURCE_URL}"
)
BAD_REPORT = (
    "# RAG\n## Summary\nA study found 40% fewer errors [1].\n"
    "## References\n[1] Invented - https://example.com/made-up"
)


def _llm(report: str, *, repaired: str | None = None) -> FakeChatModel:
    responses = [
        AIMessage(content="Plan: check memory", id="plan"),
        tool_call_message("search_memory", {"query": "retrieval augmented"}, "m1"),
        AIMessage(content="Enough evidence.", id="m2"),
        AIMessage(content=report, id="m3"),
    ]
    if repaired is not None:
        responses.append(AIMessage(content=repaired, id="m4"))
    return FakeChatModel(responses=responses)


def _result(**overrides) -> QuestionResult:
    base = {"id": "q", "category": "factual", "question": "?"}
    return QuestionResult(**{**base, **overrides})


# --- the question set ----------------------------------------------------


def test_question_set_is_usable() -> None:
    questions = load_questions()

    assert len(questions) >= 12
    assert len({q.id for q in questions}) == len(questions), "ids must be unique"
    assert {"factual", "multi-hop", "current", "hard-to-source"} <= {q.category for q in questions}
    assert all(q.question.strip().endswith("?") for q in questions)


# --- measuring one question ----------------------------------------------


def test_run_question_measures_a_clean_run(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    item = EvalQuestion(id="fact-01", question="What is RAG?", category="factual")

    result = run_question(item, test_settings, store, _llm(GOOD_REPORT))

    assert result.first_pass_valid is True
    assert result.repairs == 0
    assert result.unsupported_urls == []
    assert result.research_steps == 1
    assert result.sources >= 1
    assert result.report_chars > 0
    assert result.error is None


def test_run_question_records_a_repair(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    item = EvalQuestion(id="hard-01", question="How much does RAG help?", category="hard-to-source")

    result = run_question(item, test_settings, store, _llm(BAD_REPORT, repaired=GOOD_REPORT))

    assert result.first_pass_valid is False, "the first report cited a fabricated URL"
    assert result.repairs == 1
    assert result.unsupported_urls == [], "the repair removed it"


def test_run_question_reports_a_fabricated_url_that_survives(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    settings = test_settings.model_copy(update={"max_report_repairs": 0})
    item = EvalQuestion(id="hard-02", question="How much does RAG help?", category="hard-to-source")

    result = run_question(item, settings, store, _llm(BAD_REPORT))

    assert result.first_pass_valid is False
    assert result.repairs == 0
    assert result.unsupported_urls == ["https://example.com/made-up"]


def test_token_usage_is_summed_when_the_provider_reports_it(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    llm = _llm(GOOD_REPORT)
    for message in llm.responses:
        message.usage_metadata = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    result = run_question(EvalQuestion(id="q", question="What is RAG?"), test_settings, store, llm)

    assert result.input_tokens > 0
    assert result.output_tokens > 0


# --- aggregation and rendering -------------------------------------------


def test_aggregate_counts_what_matters() -> None:
    summary = aggregate(
        [
            _result(first_pass_valid=True, research_steps=2, sources=4, duration_s=10.0),
            _result(
                first_pass_valid=False, repairs=1, research_steps=4, sources=6, duration_s=20.0
            ),
            _result(
                first_pass_valid=False,
                repairs=1,
                unsupported_urls=["https://fake.test"],
                research_steps=3,
                sources=5,
                duration_s=30.0,
            ),
            _result(error="RuntimeError: quota exceeded", duration_s=0.0),
        ]
    )

    assert summary.questions == 4
    assert summary.completed == 3 and summary.failed == 1
    assert summary.first_pass_valid == 1
    assert summary.first_pass_valid_rate == 0.33
    assert summary.clean_after_repair == 2
    assert summary.reports_with_fabricated_urls == 1
    assert summary.total_repairs == 2
    assert summary.avg_research_steps == 3.0
    assert summary.avg_sources == 5.0
    assert summary.avg_duration_s == 15.0


def test_aggregate_handles_an_empty_run() -> None:
    summary = aggregate([])
    assert summary.questions == 0 and summary.first_pass_valid_rate == 0.0


def test_markdown_has_a_row_per_question_and_lists_failures() -> None:
    results = [
        _result(id="fact-01", first_pass_valid=True, research_steps=2, sources=3),
        _result(id="hard-04", error="TimeoutError: no response"),
    ]

    table = to_markdown(results, aggregate(results), "gemini-3.8-flash")

    assert table.count("\n| ") >= 2
    assert "| fact-01 |" in table and "| hard-04 |" in table
    assert "gemini-3.8-flash" in table
    assert "## Failures" in table and "TimeoutError" in table
