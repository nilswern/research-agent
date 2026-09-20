"""FastAPI backend of the local web UI.

A run is streamed as Server-Sent Events, so plan, progress and report appear
in the browser just as live as they do in the terminal.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.config.logging_config import get_logger
from src.config.settings import Settings, get_settings
from src.database.factory import create_store
from src.database.vector_store import VectorStore
from src.graph.runner import stream_research
from src.services.llm import create_llm
from src.services.reporting import save_report

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
REPORT_NAME = re.compile(r"^[\w.\-]+\.md$")
FRONTMATTER_QUESTION = re.compile(r'^question:\s*"?(.*?)"?\s*$', re.MULTILINE)


class ResearchRequest(BaseModel):
    """A research request coming from the browser."""

    question: str
    max_steps: int | None = Field(default=None, ge=1, le=20)
    results: int | None = Field(default=None, ge=1, le=20)
    repairs: int | None = Field(default=None, ge=0, le=3)
    language: str | None = None
    save: bool = True


def set_shutdown_hook(app: FastAPI, hook: Callable[[], None]) -> None:
    """Register how the running server can shut itself down."""
    app.state.shutdown_hook = hook


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _apply_request(settings: Settings, request: ResearchRequest) -> Settings:
    overrides: dict[str, Any] = {}
    if request.max_steps is not None:
        overrides["max_research_steps"] = request.max_steps
    if request.results is not None:
        overrides["max_results_per_search"] = request.results
    if request.repairs is not None:
        overrides["max_report_repairs"] = request.repairs
    if request.language:
        overrides["output_language"] = request.language
    overrides["stream_final_answer"] = True
    return settings.model_copy(update=overrides)


def _report_entry(path: Path) -> dict[str, Any]:
    question = ""
    try:
        head = path.read_text(encoding="utf-8")[:2000]
        match = FRONTMATTER_QUESTION.search(head)
        question = match.group(1).strip() if match else ""
    except OSError:  # pragma: no cover - unreadable file
        pass
    return {
        "name": path.name,
        "question": question or path.stem,
        "modified": path.stat().st_mtime,
    }


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    llm: BaseChatModel | None = None,
) -> FastAPI:
    """Build the FastAPI app; ``store``/``llm`` can be injected for tests."""
    settings = settings or get_settings()
    settings.ensure_directories()
    store = store if store is not None else create_store(settings)

    app = FastAPI(title="ResearchPilot", docs_url=None, redoc_url=None)
    app.state.shutdown_hook = None
    run_lock = threading.Lock()

    def _stats() -> dict[str, Any]:
        return {
            "chunks": store.count(),
            "documents": len(store.known_urls()),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chroma_path": str(settings.chroma_path),
            "reports_dir": str(settings.reports_dir),
            "defaults": {
                "max_steps": settings.max_research_steps,
                "results": settings.max_results_per_search,
                "repairs": settings.max_report_repairs,
                "language": settings.output_language,
            },
        }

    def _event_stream(request: ResearchRequest) -> Iterator[str]:
        question = request.question.strip()
        try:
            run_settings = _apply_request(settings, request)
            yield _sse({"type": "status", "text": "Research started", "level": "info"})

            state: dict[str, Any] = {}
            for event in stream_research(
                question, run_settings, store, llm=llm or create_llm(run_settings)
            ):
                if event.type == "final":
                    state = event.state or {}
                    continue
                yield _sse(
                    {
                        "type": event.type,
                        "text": event.text,
                        "node": event.node,
                        "level": event.level,
                    }
                )

            report = str(state.get("report", "")).strip()
            if not report:
                yield _sse({"type": "error", "text": "No report produced."})
                return

            saved: str | None = None
            if request.save:
                saved = save_report(
                    report=report,
                    question=question,
                    reports_dir=run_settings.reports_dir,
                    model=run_settings.llm_model,
                    research_steps=int(state.get("research_steps", 0)),
                    sources=list(state.get("sources", [])),
                ).name

            yield _sse(
                {
                    "type": "done",
                    "report": report,
                    "sources": list(state.get("sources", [])),
                    "issues": list(state.get("validation_issues", [])),
                    "research_steps": int(state.get("research_steps", 0)),
                    "repair_attempts": int(state.get("repair_attempts", 0)),
                    "saved": saved,
                    "chunks": store.count(),
                }
            )
        except Exception as exc:  # pragma: no cover - surfaced by the frontend
            logger.exception("Research run failed")
            yield _sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        finally:
            run_lock.release()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return _stats()

    @app.post("/api/purge")
    def purge() -> dict[str, Any]:
        removed = store.purge_stale()
        return {"removed": removed, **_stats()}

    @app.get("/api/reports")
    def reports() -> dict[str, Any]:
        directory = settings.reports_dir
        files = sorted(directory.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {"reports": [_report_entry(path) for path in files[:50]]}

    @app.get("/api/reports/{name}")
    def report(name: str) -> dict[str, Any]:
        if not REPORT_NAME.match(name):
            raise HTTPException(status_code=400, detail="Invalid file name.")
        path = (settings.reports_dir / name).resolve()
        if not path.is_file() or path.parent != settings.reports_dir.resolve():
            raise HTTPException(status_code=404, detail="Report not found.")
        return {"name": name, "content": path.read_text(encoding="utf-8")}

    @app.post("/api/shutdown")
    def shutdown() -> dict[str, str]:
        hook = getattr(app.state, "shutdown_hook", None)
        if hook is None:
            raise HTTPException(status_code=501, detail="This server cannot shut itself down.")
        hook()
        return {"status": "stopping"}

    @app.post("/api/research")
    def research(request: ResearchRequest) -> StreamingResponse:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="No question given.")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A research run is already in progress.")
        return StreamingResponse(
            _event_stream(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
