"""Test-Fixtures: deterministischer Fake-Embedder und Fake-LLM."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.config.settings import Settings
from src.database.vector_store import VectorStore

DIM = 16


def _fake_vector(text: str) -> list[float]:
    """Deterministischer Fake-Embedding-Vektor für Tests."""
    vector = [0.0] * DIM

    tokens = re.findall(r"\b\w+\b", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % DIM
        vector[index] += 1.0

    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _fake_vector(text)


class FakeChatModel(BaseChatModel):
    """Gibt vorgegebene Antworten der Reihe nach zurück."""

    responses: list[Any] = Field(default_factory=list)
    index: int = 0
    bound_tools: list[Any] = Field(default_factory=list)
    calls: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call_message(name: str, args: dict[str, Any], message_id: str) -> AIMessage:
    return AIMessage(
        content="",
        id=message_id,
        tool_calls=[{"name": name, "args": args, "id": f"call-{message_id}"}],
    )


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(
        path=tmp_path / "chroma",
        collection_name="test_collection",
        embedder=FakeEmbedder(),
        cache_max_age_days=30,
    )


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        google_api_key="test-key",
        max_results_per_search=3,
        max_research_steps=2,
        chunk_size=400,
        chunk_overlap=50,
        retrieval_top_k=3,
    )
