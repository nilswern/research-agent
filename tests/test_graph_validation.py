from langchain_core.messages import AIMessage

from src.graph.builder import build_graph
from src.graph.nodes import make_route_after_validation, make_validation_node
from src.graph.state import initial_state
from tests.conftest import FakeChatModel, seed_rag_document, tool_call_message

SOURCE_URL = "https://arxiv.org/abs/2005.11401"
BAD_REPORT = (
    "# RAG\n## Findings\nIt reduces hallucinations [1].\n"
    "## References\n[1] Made up - https://example.com/fake"
)
GOOD_REPORT = (
    "# RAG\n## Findings\nIt reduces hallucinations [1].\n"
    f"## References\n[1] RAG paper - {SOURCE_URL}"
)


def test_validation_node_reports_issues(store, test_settings) -> None:
    node = make_validation_node(store, test_settings)
    state = {**initial_state("q"), "report": BAD_REPORT, "sources": [SOURCE_URL]}
    assert node(state)["validation_issues"]


def test_validation_node_passes_clean_report(store, test_settings) -> None:
    node = make_validation_node(store, test_settings)
    state = {**initial_state("q"), "report": GOOD_REPORT, "sources": [SOURCE_URL]}
    assert node(state)["validation_issues"] == []


def test_route_after_validation() -> None:
    route = make_route_after_validation(max_repairs=1)
    assert route({"validation_issues": [], "repair_attempts": 0}) == "done"
    assert route({"validation_issues": ["x"], "repair_attempts": 0}) == "repair"
    assert route({"validation_issues": ["x"], "repair_attempts": 1}) == "done"


def test_graph_repairs_fabricated_source(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    llm = FakeChatModel(
        responses=[
            AIMessage(content="Plan", id="plan"),
            tool_call_message("search_memory", {"query": "rag"}, "m1"),
            AIMessage(content="Enough evidence.", id="m2"),
            AIMessage(content=BAD_REPORT, id="m3"),
            AIMessage(content=GOOD_REPORT, id="m4"),
        ]
    )
    result = build_graph(store, test_settings, llm).invoke(initial_state("What is RAG?"))

    assert result["repair_attempts"] == 1
    assert result["validation_issues"] == []
    assert "example.com/fake" not in result["report"]


def test_graph_stops_after_repair_budget(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    settings = test_settings.model_copy(update={"max_report_repairs": 1})
    llm = FakeChatModel(
        responses=[
            AIMessage(content="Plan", id="plan"),
            tool_call_message("search_memory", {"query": "rag"}, "m1"),
            AIMessage(content="Enough evidence.", id="m2"),
            AIMessage(content=BAD_REPORT, id="m3"),
            AIMessage(content=BAD_REPORT, id="m4"),
        ]
    )
    result = build_graph(store, settings, llm).invoke(initial_state("What is RAG?"))

    assert result["repair_attempts"] == 1
    assert result["validation_issues"]  # stays open and is shown to the user


def test_clean_report_skips_repair(store, test_settings) -> None:
    seed_rag_document(store, test_settings)
    llm = FakeChatModel(
        responses=[
            AIMessage(content="Plan", id="plan"),
            tool_call_message("search_memory", {"query": "rag"}, "m1"),
            AIMessage(content="Enough evidence.", id="m2"),
            AIMessage(content=GOOD_REPORT, id="m3"),
        ]
    )
    result = build_graph(store, test_settings, llm).invoke(initial_state("What is RAG?"))

    assert result["repair_attempts"] == 0
    assert result["validation_issues"] == []
