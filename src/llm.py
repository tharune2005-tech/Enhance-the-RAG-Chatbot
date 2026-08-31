"""Gemini client helpers. Matches the live-session ChatGoogleGenerativeAI setup."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, google_api_key


def extract_text(message: BaseMessage | str) -> str:
    """Gemini sometimes returns content as a string, sometimes as content blocks."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content)


def build_llm(api_key: str | None = None, model: str | None = None) -> ChatGoogleGenerativeAI:
    key = api_key or google_api_key()
    if not key:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Set it in .env, the environment, or the sidebar."
        )
    return ChatGoogleGenerativeAI(
        model=model or LLM_MODEL,
        google_api_key=key,
        temperature=LLM_TEMPERATURE,
    )


def history_as_messages(history: list[dict]) -> list[BaseMessage]:
    from langchain_core.messages import HumanMessage

    messages: list[BaseMessage] = []
    for turn in history:
        role = turn.get("role")
        text = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "assistant":
            messages.append(AIMessage(content=text))
    return messages
