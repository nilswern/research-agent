"""LLM factory (Google Gemini via langchain-google-genai)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def create_llm(settings: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.debug("Initialising LLM %s", settings.llm_model)
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        google_api_key=settings.google_api_key.get_secret_value(),
    )


class NullChatModel(BaseChatModel):
    """Placeholder for tasks that build the graph but never run it."""

    @property
    def _llm_type(self) -> str:
        return "null-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> NullChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("NullChatModel must not be called")
