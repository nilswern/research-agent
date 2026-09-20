"""Graph state and helpers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    question: str
    plan: str
    messages: Annotated[list[AnyMessage], add_messages]
    research_steps: int
    sources: list[str]
    report: str
    validation_issues: list[str]
    repair_attempts: int


def initial_state(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "plan": "",
        "messages": [],
        "research_steps": 0,
        "sources": [],
        "report": "",
        "validation_issues": [],
        "repair_attempts": 0,
    }
