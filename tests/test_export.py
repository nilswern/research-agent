from src.graph.export import graph_to_mermaid, wrap_mermaid
from src.services.llm import NullChatModel


def test_mermaid_contains_all_nodes(store, test_settings) -> None:
    mermaid = graph_to_mermaid(store, test_settings, NullChatModel())
    for node in ("plan", "agent", "tools", "synthesize", "validate", "repair"):
        assert node in mermaid


def test_mermaid_is_wrappable(store, test_settings) -> None:
    block = wrap_mermaid(graph_to_mermaid(store, test_settings, NullChatModel()))
    assert block.startswith("```mermaid")
    assert block.endswith("```")


def test_null_model_refuses_invocation() -> None:
    import pytest
    from langchain_core.messages import HumanMessage

    with pytest.raises(RuntimeError):
        NullChatModel().invoke([HumanMessage("hi")])
