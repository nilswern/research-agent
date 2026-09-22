from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.builder import build_graph
from src.graph.nodes import (
    make_plan_node,
    make_route_after_tools,
    make_tool_node,
    route_after_agent,
)
from src.graph.state import initial_state
from src.tools import build_tools
from tests.conftest import FakeChatModel, seed_rag_document, tool_call_message


def test_plan_node_seeds_messages(test_settings) -> None:
    llm = FakeChatModel(responses=[AIMessage(content="1. Check memory\n2. Search arXiv")])
    update = make_plan_node(llm, test_settings)(initial_state("What is RAG?"))

    assert "Check memory" in update["plan"]
    assert isinstance(update["messages"][0], SystemMessage)
    assert isinstance(update["messages"][1], HumanMessage)
    assert "What is RAG?" in update["messages"][1].content


def test_route_after_agent() -> None:
    with_calls = {"messages": [tool_call_message("search_memory", {"query": "x"}, "m1")]}
    without_calls = {"messages": [AIMessage(content="done")]}
    assert route_after_agent(with_calls) == "tools"
    assert route_after_agent(without_calls) == "synthesize"


def test_route_after_tools_respects_budget() -> None:
    route = make_route_after_tools(max_steps=2)
    assert route({"research_steps": 1}) == "agent"
    assert route({"research_steps": 2}) == "synthesize"


def test_tool_node_executes_and_collects_sources(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    node = make_tool_node(build_tools(store, test_settings))
    state = {
        **initial_state("What is RAG?"),
        "messages": [tool_call_message("search_memory", {"query": "retrieval augmented"}, "m1")],
    }
    update = node(state)

    assert isinstance(update["messages"][0], ToolMessage)
    assert update["research_steps"] == 1
    assert "https://arxiv.org/abs/2005.11401" in update["sources"]


def test_tool_node_handles_unknown_tool(store, test_settings) -> None:
    node = make_tool_node(build_tools(store, test_settings))
    state = {
        **initial_state("q"),
        "messages": [tool_call_message("does_not_exist", {}, "m1")],
    }
    assert "Unknown tool" in node(state)["messages"][0].content


def test_tool_node_survives_failing_tool(store, test_settings, monkeypatch) -> None:
    from src.tools import web_search as web_search_module

    def failing(query: str, max_results: int):
        raise RuntimeError("Web search unavailable: rate limit")

    monkeypatch.setattr(web_search_module, "run_web_search", failing)
    node = make_tool_node(build_tools(store, test_settings))
    state = {
        **initial_state("q"),
        "messages": [tool_call_message("web_search", {"query": "x"}, "m1")],
    }
    output = node(state)["messages"][0].content
    assert "unavailable" in output.lower()
    assert output.lower().count("web search unavailable") == 1, "no doubled prefix"


def test_full_run_with_one_tool_round(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    llm = FakeChatModel(
        responses=[
            AIMessage(content="Plan: check memory", id="plan"),
            tool_call_message("search_memory", {"query": "retrieval augmented"}, "m1"),
            AIMessage(content="Enough evidence collected.", id="m2"),
            AIMessage(
                content="# RAG\n## Summary\nIt combines retrieval and generation [1].\n"
                "## References\n[1] RAG - https://arxiv.org/abs/2005.11401",
                id="m3",
            ),
        ]
    )
    graph = build_graph(store, test_settings, llm)
    result = graph.invoke(initial_state("What is RAG?"))

    assert result["research_steps"] == 1
    assert "https://arxiv.org/abs/2005.11401" in result["sources"]
    assert result["report"].startswith("# RAG")


def test_budget_stops_endless_loop(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    llm = FakeChatModel(
        responses=[
            AIMessage(content="Plan", id="plan"),
            tool_call_message("search_memory", {"query": "a"}, "m1"),
            tool_call_message("search_memory", {"query": "b"}, "m2"),
            AIMessage(content="# Report\n## Summary\nDone.", id="final"),
        ]
    )
    graph = build_graph(store, test_settings, llm)
    result = graph.invoke(initial_state("What is RAG?"))

    assert result["research_steps"] == test_settings.max_research_steps == 2
    assert result["report"].startswith("# Report")


def test_tools_are_bound_to_llm(store, test_settings) -> None:
    llm = FakeChatModel(responses=[AIMessage(content="x")])
    build_graph(store, test_settings, llm)
    assert {tool.name for tool in llm.bound_tools} == {
        "search_memory",
        "web_search",
        "wikipedia_search",
        "arxiv_search",
        "scrape_webpage",
    }
