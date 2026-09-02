"""Groq client wrapper for fast LLM calls (intent classification, grading)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import settings

if TYPE_CHECKING:
    from groq import Groq

_client: Groq | None = None


def get_groq_client() -> Groq:
    from groq import Groq

    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.3) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model or settings.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
