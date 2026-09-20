"""Assembly of the LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.graph.nodes import (
    make_agent_node,
    make_plan_node,
    make_repair_node,
    make_route_after_tools,
    make_route_after_validation,
    make_synthesis_node,
    make_tool_node,
    route_after_agent,
    validation_node,
)
from src.graph.state import ResearchState
from src.tools import build_tools


def build_graph(store: VectorStore, settings: Settings, llm: BaseChatModel) -> CompiledStateGraph:
    tools = build_tools(store, settings)

    # dict[str, Any] because the add_node overloads cannot resolve callables
    # that come out of a factory function.
    nodes: dict[str, Any] = {
        "plan": make_plan_node(llm, settings),
        "agent": make_agent_node(llm.bind_tools(tools)),
        "tools": make_tool_node(tools),
        "synthesize": make_synthesis_node(llm, store, settings),
        "validate": validation_node,
        "repair": make_repair_node(llm),
    }

    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "synthesize": "synthesize"}
    )
    graph.add_conditional_edges(
        "tools",
        make_route_after_tools(settings.max_research_steps),
        {"agent": "agent", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate",
        make_route_after_validation(settings.max_report_repairs),
        {"repair": "repair", "done": END},
    )
    graph.add_edge("repair", "validate")

    return graph.compile()


def recursion_limit(settings: Settings) -> int:
    return settings.max_research_steps * 2 + settings.max_report_repairs * 2 + 10
