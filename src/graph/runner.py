"""Oberflächenunabhängiger Event-Stream eines Research-Laufs.

CLI und Web-UI konsumieren dieselben Events, damit die Fortschrittsanzeige
in beiden Frontends identisch bleibt.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.graph.builder import build_graph, recursion_limit
from src.graph.nodes import as_text
from src.graph.state import initial_state
from src.services.llm import create_llm

STATE_KEYS = (
    "plan",
    "report",
    "sources",
    "research_steps",
    "validation_issues",
    "repair_attempts",
)
STREAMING_NODES = {"synthesize": "Report", "repair": "Corrected report"}


@dataclass(slots=True)
class ResearchEvent:
    """Ein Ereignis während eines Laufs.

    ``type`` ist eines von:
    ``plan`` (Rechercheplan), ``status`` (Fortschrittszeile),
    ``token`` (Report-Fragment beim Streaming), ``report`` (kompletter Report
    ohne Streaming) und ``final`` (Endzustand des Graphen in ``state``).
    """

    type: str
    text: str = ""
    node: str = ""
    level: str = "info"  # info | ok | warn
    state: dict[str, Any] | None = None


def merge_state(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key in STATE_KEYS:
        if key in update and update[key] is not None:
            state[key] = update[key]


def describe_update(node: str, update: dict[str, Any]) -> tuple[str, str] | None:
    """Übersetzt ein Node-Update in eine Fortschrittszeile mit Level."""
    if node == "plan":
        return "Plan erstellt", "info"
    if node == "agent":
        messages = update.get("messages") or []
        calls = getattr(messages[-1], "tool_calls", None) if messages else None
        if calls:
            return "→ " + ", ".join(call["name"] for call in calls), "info"
        return "→ genug Belege gesammelt", "info"
    if node == "tools":
        rounds = (
            f"Runde {update.get('research_steps', '?')} · {len(update.get('sources', []))} Quellen"
        )
        return rounds, "info"
    if node == "synthesize":
        return "Report wird geschrieben", "info"
    if node == "validate":
        issues = update.get("validation_issues") or []
        if issues:
            return f"Validierung: {len(issues)} Problem(e), Repair läuft", "warn"
        return "Validierung bestanden", "ok"
    return None


def stream_research(
    question: str,
    settings: Settings,
    store: VectorStore,
    llm: BaseChatModel | None = None,
) -> Iterator[ResearchEvent]:
    """Führt einen Research-Lauf aus und liefert dabei Events."""
    graph = build_graph(store, settings, llm or create_llm(settings))
    state = initial_state(question)
    config: RunnableConfig = {"recursion_limit": recursion_limit(settings)}

    if not settings.stream_final_answer:
        result = dict(graph.invoke(state, config=config))
        if result.get("plan"):
            yield ResearchEvent("plan", text=result["plan"], node="plan")
        yield ResearchEvent("report", text=result.get("report", ""), node="synthesize")
        yield ResearchEvent("final", state=result)
        return

    streamed: list[str] = []
    streaming_node: str | None = None

    # cast: mit mehreren stream_modes liefert LangGraph (mode, payload)-Tupel,
    # die Signatur gibt das aber nicht her.
    events = cast(
        Iterator[tuple[str, Any]],
        graph.stream(state, config=config, stream_mode=["updates", "messages"]),
    )

    for mode, payload in events:
        if mode == "updates":
            for node, update in (payload or {}).items():
                if not isinstance(update, dict):
                    continue
                merge_state(state, update)
                described = describe_update(node, update)
                if described:
                    text, level = described
                    yield ResearchEvent("status", text=text, node=node, level=level)
                if node == "plan" and update.get("plan"):
                    yield ResearchEvent("plan", text=update["plan"], node=node)
        elif mode == "messages":
            chunk, metadata = payload
            node = metadata.get("langgraph_node")
            if node not in STREAMING_NODES:
                continue
            text = as_text(chunk.content)
            if not text:
                continue
            if node != streaming_node:
                streaming_node = node
                streamed = []
            streamed.append(text)
            yield ResearchEvent("token", text=text, node=node)

    if streamed and not state.get("report"):
        state["report"] = "".join(streamed).strip()

    yield ResearchEvent("final", state=state)
