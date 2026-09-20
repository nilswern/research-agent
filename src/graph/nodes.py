"""Node-Implementierungen des Research-Graphen."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from src.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_TEMPLATE,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_TASK_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_TEMPLATE,
)
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.graph.state import ResearchState
from src.services.validation import extract_urls, validate_report

logger = get_logger(__name__)


def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return "".join(parts)
    return str(content)


def find_unsupported_urls(report: str, allowed: list[str]) -> list[str]:
    allowed_set = set(allowed)
    return [url for url in extract_urls(report) if url not in allowed_set]


# --- Nodes ---------------------------------------------------------------


def make_plan_node(
    llm: BaseChatModel, settings: Settings
) -> Callable[[ResearchState], dict[str, Any]]:
    def plan_node(state: ResearchState) -> dict[str, Any]:
        question = state["question"]
        response = llm.invoke([SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(question)])
        plan = as_text(response.content).strip()
        logger.info("Plan erstellt (%d Zeichen)", len(plan))

        return {
            "plan": plan,
            "messages": [
                SystemMessage(RESEARCH_SYSTEM_PROMPT.format(max_steps=settings.max_research_steps)),
                HumanMessage(RESEARCH_TASK_TEMPLATE.format(question=question, plan=plan)),
            ],
        }

    return plan_node


def make_agent_node(
    llm_with_tools: Runnable[Any, Any],
) -> Callable[[ResearchState], dict[str, Any]]:
    def agent_node(state: ResearchState) -> dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            logger.info(
                "Agent fordert Tools an: %s",
                ", ".join(call["name"] for call in tool_calls),
            )
        else:
            logger.info("Agent beendet die Recherche")
        return {"messages": [response]}

    return agent_node


def make_tool_node(tools: list[BaseTool]) -> Callable[[ResearchState], dict[str, Any]]:
    registry = {tool.name: tool for tool in tools}

    def tool_node(state: ResearchState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []

        outputs: list[ToolMessage] = []
        sources = list(state.get("sources", []))

        for call in tool_calls:
            name = call["name"]
            tool = registry.get(name)
            if tool is None:
                content = f"Unknown tool '{name}'. Available: {', '.join(registry)}."
                logger.warning(content)
            else:
                try:
                    content = as_text(tool.invoke(call.get("args", {})))
                except Exception as exc:
                    logger.warning("Tool '%s' failed: %s", name, exc)
                    content = f"Tool '{name}' failed: {exc}. Try a different approach."

            for url in extract_urls(content):
                if url not in sources:
                    sources.append(url)

            outputs.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))

        step = state.get("research_steps", 0) + 1
        logger.info("Research-Runde %d abgeschlossen, %d Quellen bekannt", step, len(sources))
        return {"messages": outputs, "research_steps": step, "sources": sources}

    return tool_node


def make_synthesis_node(
    llm: BaseChatModel,
    store: VectorStore,
    settings: Settings,
) -> Callable[[ResearchState], dict[str, Any]]:
    def synthesis_node(state: ResearchState) -> dict[str, Any]:
        hits = store.query(state["question"], top_k=settings.retrieval_top_k * 2)
        allowed = list(state.get("sources", []))

        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            if hit.url and hit.url not in allowed:
                allowed.append(hit.url)
            blocks.append(f"[{index}] {hit.title} - {hit.url}\n{hit.text}")

        prompt = SYNTHESIS_TEMPLATE.format(
            question=state["question"],
            context="\n\n".join(blocks) or "(no stored passages)",
            sources="\n".join(f"- {url}" for url in allowed) or "(none)",
            language=settings.output_language,
        )

        response = llm.invoke(
            [*state["messages"], SystemMessage(SYNTHESIS_SYSTEM_PROMPT), HumanMessage(prompt)]
        )
        report = as_text(response.content).strip()
        logger.info("Report erstellt (%d Zeichen, %d Quellen)", len(report), len(allowed))
        return {"report": report, "sources": allowed, "messages": [AIMessage(report)]}

    return synthesis_node


def validation_node(state: ResearchState) -> dict[str, Any]:
    result = validate_report(state.get("report", ""), list(state.get("sources", [])))
    issues = result.as_issues()
    if issues:
        logger.warning("Validierung fehlgeschlagen: %s", " | ".join(issues))
    else:
        logger.info("Validierung bestanden")
    return {"validation_issues": issues}


def make_repair_node(llm: BaseChatModel) -> Callable[[ResearchState], dict[str, Any]]:
    def repair_node(state: ResearchState) -> dict[str, Any]:
        prompt = REPAIR_TEMPLATE.format(
            issues="\n".join(f"- {issue}" for issue in state["validation_issues"]),
            sources="\n".join(f"- {url}" for url in state.get("sources", [])) or "(none)",
            report=state["report"],
        )
        response = llm.invoke([SystemMessage(REPAIR_SYSTEM_PROMPT), HumanMessage(prompt)])
        attempts = state.get("repair_attempts", 0) + 1
        logger.info("Repair-Runde %d durchgeführt", attempts)
        return {"report": as_text(response.content).strip(), "repair_attempts": attempts}

    return repair_node


# --- Routing -------------------------------------------------------------


def route_after_agent(state: ResearchState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "synthesize"


def make_route_after_tools(max_steps: int) -> Callable[[ResearchState], str]:
    def route_after_tools(state: ResearchState) -> str:
        if state.get("research_steps", 0) >= max_steps:
            logger.info("Research-Budget (%d Runden) aufgebraucht", max_steps)
            return "synthesize"
        return "agent"

    return route_after_tools


def make_route_after_validation(max_repairs: int) -> Callable[[ResearchState], str]:
    def route_after_validation(state: ResearchState) -> str:
        if not state.get("validation_issues"):
            return "done"
        if state.get("repair_attempts", 0) >= max_repairs:
            logger.warning(
                "Repair-Budget aufgebraucht, Report bleibt mit %d offenen Problemen",
                len(state["validation_issues"]),
            )
            return "done"
        return "repair"

    return route_after_validation
