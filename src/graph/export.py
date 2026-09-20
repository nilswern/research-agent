"""Exports the graph structure as a Mermaid diagram."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config.settings import Settings
from src.database.vector_store import VectorStore
from src.graph.builder import build_graph


def graph_to_mermaid(store: VectorStore, settings: Settings, llm: BaseChatModel) -> str:
    """Mermaid source of the compiled graph."""
    return build_graph(store, settings, llm).get_graph().draw_mermaid().strip()


def wrap_mermaid(source: str) -> str:
    return f"```mermaid\n{source}\n```"
