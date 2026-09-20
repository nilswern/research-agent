import os
from datetime import UTC
from pathlib import Path

from langchain_core.messages import AIMessage
from rich.console import Console

from src.cli import apply_overrides, parse_args, run_research
from src.models.document import SourceDocument, SourceTool
from src.services.reporting import build_frontmatter, save_report, slugify
from tests.conftest import FakeChatModel, tool_call_message

REPORT = (
    "# RAG\n## Summary\nIt combines retrieval and generation [1].\n"
    "## References\n[1] RAG - https://arxiv.org/abs/2005.11401"
)


def _console() -> Console:
    return Console(
        file=open(os.devnull, "w", encoding="utf-8"),
        force_terminal=False,
    )


def _seed(store, settings) -> None:
    store.add_document(
        SourceDocument(
            source_tool=SourceTool.ARXIV,
            url="https://arxiv.org/abs/2005.11401",
            title="Retrieval Augmented Generation",
            text="Retrieval augmented generation combines retrieval and generation. " * 20,
        ),
        settings.chunk_size,
        settings.chunk_overlap,
    )


def _llm() -> FakeChatModel:
    return FakeChatModel(
        responses=[
            AIMessage(content="Plan: check memory", id="plan"),
            tool_call_message("search_memory", {"query": "retrieval augmented"}, "m1"),
            AIMessage(content="Enough evidence.", id="m2"),
            AIMessage(content=REPORT, id="m3"),
        ]
    )


def test_slugify_handles_umlauts_and_punctuation() -> None:
    assert slugify("Wie funktioniert RAG? Überblick!") == "wie-funktioniert-rag-uberblick"
    assert slugify("???") == "report"


def test_parse_args_defaults() -> None:
    args = parse_args(["what", "is", "rag"])
    assert args.question == ["what", "is", "rag"]
    assert args.max_steps is None and args.no_stream is False


def test_apply_overrides(test_settings) -> None:
    args = parse_args(["q", "--max-steps", "9", "--language", "de", "--no-stream"])
    updated = apply_overrides(test_settings, args)
    assert updated.max_research_steps == 9
    assert updated.output_language == "de"
    assert updated.stream_final_answer is False
    assert test_settings.max_research_steps == 2  # original untouched


def test_frontmatter_contains_sources() -> None:
    from datetime import datetime

    text = build_frontmatter(
        "What is RAG?", "gemini-3.8-flash", 2, ["https://a.test"], datetime.now(UTC)
    )
    assert text.startswith("---")
    assert "  - https://a.test" in text
    assert "research_steps: 2" in text


def test_save_report_writes_file(tmp_path: Path) -> None:
    path = save_report(
        report=REPORT,
        question="What is RAG?",
        reports_dir=tmp_path / "reports",
        model="gemini-3.8-flash",
        research_steps=1,
        sources=["https://arxiv.org/abs/2005.11401"],
    )
    content = path.read_text(encoding="utf-8")
    assert path.suffix == ".md"
    assert "what-is-rag" in path.name
    assert content.startswith("---")
    assert "# RAG" in content


def test_run_research_without_streaming(store, test_settings) -> None:
    _seed(store, test_settings)
    settings = test_settings.model_copy(update={"stream_final_answer": False})
    result = run_research("What is RAG?", settings, store, _console(), llm=_llm())

    assert result["report"].startswith("# RAG")
    assert result["research_steps"] == 1
    assert "https://arxiv.org/abs/2005.11401" in result["sources"]


def test_run_research_with_streaming(store, test_settings) -> None:
    _seed(store, test_settings)
    result = run_research("What is RAG?", test_settings, store, _console(), llm=_llm())

    assert "# RAG" in result["report"]
    assert result["research_steps"] == 1
    assert result["plan"].startswith("Plan")


def test_parse_args_graph_flag() -> None:
    assert parse_args(["--graph"]).graph is True


def test_apply_overrides_repairs(test_settings) -> None:
    updated = apply_overrides(test_settings, parse_args(["q", "--repairs", "3"]))
    assert updated.max_report_repairs == 3
