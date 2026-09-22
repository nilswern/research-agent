"""Run the agent over a question set and measure how well it sources its answers.

    python -m evals.run_eval --limit 3

This makes real LLM and network calls, so it is never part of CI. It writes a
JSON file with every measurement and a Markdown table for the README.

What is measured per question, and why:

first_pass_valid   Did the first report survive validation without a repair?
                   The honest headline number - repairs hide fabrications.
repairs            How often the repair stage had to run.
unsupported_urls   URLs in the report that no tool ever returned, after all
                   repairs. Anything above zero shipped a fabricated source.
dangling           Citation markers without a reference entry.
unsupported_claims Sentences whose cited source does not back them up
                   (only when FAITHFULNESS_CHECK is on).
steps / sources    How much research the agent actually did.
duration / tokens  What it cost.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import Settings, get_settings
from src.database.factory import create_store
from src.database.vector_store import VectorStore
from src.graph.builder import build_graph, recursion_limit
from src.graph.state import initial_state
from src.services.faithfulness import make_support_checker
from src.services.llm import create_llm
from src.services.validation import validate_report

logger = get_logger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: str = "uncategorised"
    probes: str = ""


class QuestionResult(BaseModel):
    id: str
    category: str
    question: str
    first_pass_valid: bool = False
    repairs: int = 0
    unsupported_urls: list[str] = Field(default_factory=list)
    dangling_citations: list[int] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    research_steps: int = 0
    sources: int = 0
    report_chars: int = 0
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Summary(BaseModel):
    questions: int = 0
    completed: int = 0
    failed: int = 0
    first_pass_valid: int = 0
    first_pass_valid_rate: float = 0.0
    clean_after_repair: int = 0
    reports_with_fabricated_urls: int = 0
    total_repairs: int = 0
    avg_research_steps: float = 0.0
    avg_sources: float = 0.0
    avg_duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(EvalQuestion.model_validate_json(line))
    return questions


def token_usage(messages: list[Any]) -> tuple[int, int]:
    """Sum token usage over the run; providers that report none give (0, 0)."""
    inputs = outputs = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            inputs += int(usage.get("input_tokens", 0) or 0)
            outputs += int(usage.get("output_tokens", 0) or 0)
    return inputs, outputs


def run_question(
    item: EvalQuestion,
    settings: Settings,
    store: VectorStore,
    llm: BaseChatModel | None = None,
) -> QuestionResult:
    """Run one question end to end and measure it."""
    result = QuestionResult(id=item.id, category=item.category, question=item.question)
    graph = build_graph(store, settings, llm or create_llm(settings))
    config: RunnableConfig = {"recursion_limit": recursion_limit(settings)}

    first_validation: list[str] | None = None
    final: dict[str, Any] = {}
    started = time.perf_counter()

    try:
        events = cast(
            Iterator[tuple[str, Any]],
            graph.stream(
                initial_state(item.question),
                config=config,
                stream_mode=["updates", "values"],
            ),
        )
        for mode, payload in events:
            if mode == "values" and isinstance(payload, dict):
                final = payload
            elif mode == "updates":
                for node, update in (payload or {}).items():
                    if node == "validate" and first_validation is None and isinstance(update, dict):
                        first_validation = list(update.get("validation_issues") or [])
    except Exception as exc:  # a failed run is a measurement, not a crash
        logger.warning("%s failed: %s", item.id, exc)
        result.error = f"{type(exc).__name__}: {exc}"

    result.duration_s = round(time.perf_counter() - started, 2)
    if result.error:
        return result

    report = str(final.get("report", ""))
    sources = list(final.get("sources", []))
    final_check = validate_report(report, sources, make_support_checker(store, settings))

    result.first_pass_valid = first_validation == []
    result.repairs = int(final.get("repair_attempts", 0))
    result.unsupported_urls = final_check.unsupported_urls
    result.dangling_citations = final_check.dangling_citations
    result.unsupported_claims = final_check.unsupported_claims
    result.research_steps = int(final.get("research_steps", 0))
    result.sources = len(sources)
    result.report_chars = len(report)
    result.input_tokens, result.output_tokens = token_usage(list(final.get("messages", [])))
    return result


def aggregate(results: list[QuestionResult]) -> Summary:
    summary = Summary(questions=len(results))
    if not results:
        return summary

    completed = [r for r in results if r.error is None]
    summary.completed = len(completed)
    summary.failed = len(results) - len(completed)
    summary.total_repairs = sum(r.repairs for r in results)
    summary.input_tokens = sum(r.input_tokens for r in results)
    summary.output_tokens = sum(r.output_tokens for r in results)
    summary.avg_duration_s = round(sum(r.duration_s for r in results) / len(results), 2)

    if completed:
        summary.first_pass_valid = sum(1 for r in completed if r.first_pass_valid)
        summary.first_pass_valid_rate = round(summary.first_pass_valid / len(completed), 2)
        summary.clean_after_repair = sum(
            1 for r in completed if not (r.unsupported_urls or r.dangling_citations)
        )
        summary.reports_with_fabricated_urls = sum(1 for r in completed if r.unsupported_urls)
        summary.avg_research_steps = round(
            sum(r.research_steps for r in completed) / len(completed), 2
        )
        summary.avg_sources = round(sum(r.sources for r in completed) / len(completed), 2)
    return summary


def to_markdown(results: list[QuestionResult], summary: Summary, model: str) -> str:
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Evaluation run",
        "",
        f"Model `{model}` · {created} · {summary.questions} questions",
        "",
        f"- First report valid without repair: **{summary.first_pass_valid}/{summary.completed}**"
        f" ({summary.first_pass_valid_rate:.0%})",
        f"- Clean after the repair budget: **{summary.clean_after_repair}/{summary.completed}**",
        f"- Reports still containing a fabricated URL: **{summary.reports_with_fabricated_urls}**",
        f"- Average research rounds: {summary.avg_research_steps} ·"
        f" sources: {summary.avg_sources} · duration: {summary.avg_duration_s}s",
        f"- Tokens: {summary.input_tokens} in / {summary.output_tokens} out",
        "",
        "| Question | Category | 1st pass | Repairs | Bad URLs | Dangling | Claims |"
        " Steps | Sources | Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        status = "error" if r.error else ("yes" if r.first_pass_valid else "no")
        lines.append(
            f"| {r.id} | {r.category} | {status} | {r.repairs} |"
            f" {len(r.unsupported_urls)} | {len(r.dangling_citations)} |"
            f" {len(r.unsupported_claims)} | {r.research_steps} | {r.sources} | {r.duration_s} |"
        )

    failures = [r for r in results if r.error]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r.id}`: {r.error}" for r in failures]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals.run_eval",
        description="Measure source quality over the eval question set (real API calls).",
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--category", default=None, help="only this category")
    parser.add_argument("--max-steps", type=int, default=None, help="override MAX_RESEARCH_STEPS")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings().model_copy(update={"stream_final_answer": False})
    if args.max_steps is not None:
        settings = settings.model_copy(update={"max_research_steps": args.max_steps})

    setup_logging(settings.log_level if args.verbose else "WARNING")
    settings.ensure_directories()

    questions = load_questions(args.questions)
    if args.category:
        questions = [q for q in questions if q.category == args.category]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        print("No questions selected.")
        return 2

    store = create_store(settings)
    llm = create_llm(settings)

    results: list[QuestionResult] = []
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item.id}: {item.question}")
        result = run_question(item, settings, store, llm)
        results.append(result)
        print(
            f"    first pass valid: {result.first_pass_valid} · repairs: {result.repairs}"
            f" · bad URLs: {len(result.unsupported_urls)} · {result.duration_s}s"
        )

    summary = aggregate(results)
    markdown = to_markdown(results, summary, settings.llm_model)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = args.out / f"{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "model": settings.llm_model,
                "created_at": datetime.now(UTC).isoformat(),
                "settings": {
                    "max_research_steps": settings.max_research_steps,
                    "max_results_per_search": settings.max_results_per_search,
                    "max_report_repairs": settings.max_report_repairs,
                    "faithfulness_check": settings.faithfulness_check,
                },
                "summary": summary.model_dump(),
                "results": [r.model_dump() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / f"{stamp}.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
