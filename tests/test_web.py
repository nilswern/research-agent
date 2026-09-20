"""Tests der Web-API - ohne Netzwerk, mit Fake-LLM."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.models.document import SourceDocument, SourceTool
from src.web.app import create_app, set_shutdown_hook
from tests.conftest import FakeChatModel, tool_call_message

REPORT = (
    "# RAG\n## Summary\nIt combines retrieval and generation [1].\n"
    "## References\n[1] RAG - https://arxiv.org/abs/2005.11401"
)


def _llm() -> FakeChatModel:
    return FakeChatModel(
        responses=[
            AIMessage(content="Plan: check memory", id="plan"),
            tool_call_message("search_memory", {"query": "retrieval augmented"}, "m1"),
            AIMessage(content="Enough evidence.", id="m2"),
            AIMessage(content=REPORT, id="m3"),
        ]
    )


@pytest.fixture
def client(store, test_settings, tmp_path) -> TestClient:
    settings = test_settings.model_copy(update={"reports_dir": tmp_path / "reports"})
    store.add_document(
        SourceDocument(
            source_tool=SourceTool.ARXIV,
            url="https://arxiv.org/abs/2005.11401",
            title="Retrieval Augmented Generation",
            text="Retrieval augmented generation combines retrieval and generation. " * 20,
        ),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return TestClient(create_app(settings=settings, store=store, llm=_llm()))


def _events(response) -> list[dict]:
    return [
        json.loads(block.split("data: ", 1)[1])
        for block in response.text.split("\n\n")
        if block.startswith("data: ")
    ]


def test_index_serves_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ResearchPilot" in response.text


def test_stats_reports_memory(client: TestClient) -> None:
    body = client.get("/api/stats").json()
    assert body["chunks"] > 0
    assert body["defaults"]["max_steps"] == 2


def test_research_streams_events_and_saves_report(client: TestClient, tmp_path) -> None:
    response = client.post("/api/research", json={"question": "What is RAG?"})
    assert response.status_code == 200

    events = _events(response)
    types = [event["type"] for event in events]
    assert "plan" in types and "status" in types
    assert types[-1] == "done"

    done = events[-1]
    assert "# RAG" in done["report"]
    assert done["research_steps"] == 1
    assert "https://arxiv.org/abs/2005.11401" in done["sources"]
    assert done["saved"].endswith(".md")
    assert (tmp_path / "reports" / done["saved"]).is_file()

    listed = client.get("/api/reports").json()["reports"]
    assert listed[0]["question"] == "What is RAG?"
    assert "# RAG" in client.get(f"/api/reports/{done['saved']}").json()["content"]


def test_research_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/api/research", json={"question": "   "}).status_code == 400


def test_report_download_rejects_traversal(client: TestClient) -> None:
    assert client.get("/api/reports/..%2F..%2F.env").status_code in {400, 404}


def test_shutdown_needs_a_hook(client: TestClient) -> None:
    assert client.post("/api/shutdown").status_code == 501


def test_shutdown_calls_the_hook(client: TestClient) -> None:
    called: list[bool] = []
    set_shutdown_hook(client.app, lambda: called.append(True))

    assert client.post("/api/shutdown").json() == {"status": "stopping"}
    assert called == [True]
