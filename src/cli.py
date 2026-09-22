"""Command line entry point for ResearchPilot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import Settings, get_settings
from src.database.factory import create_store
from src.database.vector_store import VectorStore
from src.graph.export import graph_to_mermaid, wrap_mermaid
from src.graph.runner import STREAMING_NODES, stream_research
from src.services.llm import NullChatModel
from src.services.reporting import save_report

logger = get_logger(__name__)

STATUS_STYLES = {"ok": "green", "warn": "yellow"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="researchpilot",
        description="Autonomous research agent (LangGraph + Gemini + ChromaDB).",
    )
    parser.add_argument("question", nargs="*", help="research question")
    parser.add_argument("--max-steps", type=int, default=None, help="override MAX_RESEARCH_STEPS")
    parser.add_argument("--results", type=int, default=None, help="override MAX_RESULTS_PER_SEARCH")
    parser.add_argument("--repairs", type=int, default=None, help="override MAX_REPORT_REPAIRS")
    parser.add_argument("--language", default=None, help="output language, e.g. en or de")
    parser.add_argument("--no-stream", action="store_true", help="disable streaming")
    parser.add_argument("--no-save", action="store_true", help="do not write a report file")
    parser.add_argument("--purge", action="store_true", help="drop stale chunks before running")
    parser.add_argument("--stats", action="store_true", help="show memory stats and exit")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    parser.add_argument("--verbose", action="store_true", help="show agent logs")
    parser.add_argument("--web", action="store_true", help="start the local browser UI")
    parser.add_argument("--host", default="127.0.0.1", help="host for --web")
    parser.add_argument("--port", type=int, default=8000, help="port for --web")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides["max_research_steps"] = args.max_steps
    if args.results is not None:
        overrides["max_results_per_search"] = args.results
    if args.repairs is not None:
        overrides["max_report_repairs"] = args.repairs
    if args.language is not None:
        overrides["output_language"] = args.language
    if args.no_stream:
        overrides["stream_final_answer"] = False
    return settings.model_copy(update=overrides) if overrides else settings


def show_stats(store: VectorStore, settings: Settings, console: Console) -> None:
    table = Table(title="ResearchPilot memory", show_header=False)
    table.add_row("Chunks", str(store.count()))
    table.add_row("Documents (unique URLs)", str(len(store.known_urls())))
    table.add_row("ChromaDB path", str(settings.chroma_path))
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("LLM", settings.llm_model)
    table.add_row("Max research steps", str(settings.max_research_steps))
    console.print(table)


def print_issues(issues: list[str], console: Console) -> None:
    if not issues:
        return
    console.print(
        Panel(
            "\n".join(f"• {issue}" for issue in issues),
            title="[yellow]Unresolved source issues[/yellow]",
            expand=False,
        )
    )


def run_research(
    question: str,
    settings: Settings,
    store: VectorStore,
    console: Console,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    streaming_node: str | None = None

    for event in stream_research(question, settings, store, llm=llm):
        if event.type == "token":
            if event.node != streaming_node:
                streaming_node = event.node
                console.print(Rule(STREAMING_NODES.get(event.node, "Report")))
            console.print(event.text, end="", markup=False, highlight=False)
            continue

        if streaming_node:
            console.print()
            streaming_node = None

        if event.type == "final":
            state = event.state or {}
        elif event.type == "plan":
            console.print(Panel(event.text, title="Research plan", expand=False))
        elif event.type == "report":
            console.print(Rule("Report"))
            console.print(Markdown(event.text))
        elif event.type == "status":
            style = STATUS_STYLES.get(event.level)
            console.print(f"[{style}]{event.text}[/{style}]" if style else event.text)

    if streaming_node:
        console.print()
    return state


def serve_web(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    """Start the local web UI."""
    try:
        from src.web.server import run_server
    except ModuleNotFoundError as exc:  # fastapi/uvicorn missing
        console.print(f"[red]Web UI unavailable:[/red] {exc}")
        console.print("Install the dependencies: pip install -r requirements.txt")
        return 2

    url = f"http://{args.host}:{args.port}"
    console.print(f"[green]ResearchPilot UI:[/green] {url}  [dim](Ctrl+C stops it)[/dim]")
    try:
        run_server(
            settings,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            log_level=settings.log_level if args.verbose else "warning",
        )
    except OSError as exc:
        console.print(f"[red]Server failed to start:[/red] {exc}")
        console.print(f"Is ResearchPilot already running on port {args.port}?")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    try:
        settings = apply_overrides(get_settings(), args)
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        console.print("Copy .env.example to .env and set GOOGLE_API_KEY.")
        return 2

    setup_logging(settings.log_level if args.verbose else "WARNING")
    settings.ensure_directories()

    if args.web:
        return serve_web(args, settings, console)

    store = create_store(settings)

    if args.purge:
        console.print(f"{store.purge_stale()} stale chunks removed")

    if args.graph:
        console.print(
            wrap_mermaid(graph_to_mermaid(store, settings, NullChatModel())),
            markup=False,
            highlight=False,
        )
        return 0

    if args.stats:
        show_stats(store, settings, console)
        return 0

    question = (
        " ".join(args.question).strip() or console.input("[bold]Research question:[/bold] ").strip()
    )
    if not question:
        console.print("[red]No question given.[/red]")
        return 2

    console.print(Panel(question, title="Question", expand=False))

    try:
        result = run_research(question, settings, store, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    except Exception as exc:
        logger.exception("Research run failed")
        console.print(f"[red]Research failed:[/red] {exc}")
        return 1

    report = result.get("report", "")
    if not report:
        console.print("[red]No report produced.[/red]")
        return 1

    print_issues(list(result.get("validation_issues", [])), console)

    if not args.no_save:
        path: Path = save_report(
            report=report,
            question=question,
            reports_dir=settings.reports_dir,
            model=settings.llm_model,
            research_steps=int(result.get("research_steps", 0)),
            sources=list(result.get("sources", [])),
        )
        console.print(f"\n[green]Saved:[/green] {path}")

    console.print(
        f"[dim]{result.get('research_steps', 0)} rounds · "
        f"{len(result.get('sources', []))} sources · "
        f"{result.get('repair_attempts', 0)} repairs · "
        f"{store.count()} chunks in memory[/dim]"
    )
    return 0
