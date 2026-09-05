"""Groq client wrapper for fast LLM calls (intent classification, grading) with key rotation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.config import settings

if TYPE_CHECKING:
    from groq import Groq

logger = logging.getLogger("ai.groq_client")


def get_groq_client(api_key: str | None = None) -> Groq:
    from groq import Groq
    from src.llm.key_rotator import AIKeyRotator

    if not api_key:
        try:
            api_key = AIKeyRotator.get_instance().get_next_key(provider="groq")
        except Exception:
            api_key = settings.GROQ_API_KEY

    return Groq(api_key=api_key or settings.GROQ_API_KEY)


def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.3) -> str:
    """Execute chat completion with automatic Groq key rotation and failover."""
    from src.llm.key_rotator import AIKeyRotator

    rotator = AIKeyRotator.get_instance()
    max_retries = 3

    for attempt in range(max_retries):
        current_key = None
        try:
            current_key = rotator.get_next_key(provider="groq")
        except Exception:
            current_key = settings.GROQ_API_KEY

        if not current_key:
            logger.warning("No Groq API key available in rotator or settings.")
            return ""

        try:
            client = get_groq_client(api_key=current_key)
            response = client.chat.completions.create(
                model=model or settings.GROQ_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                logger.warning("Groq rate limit hit on key. Placing in cooldown and retrying...")
                rotator.report_rate_limit(current_key, cooldown_seconds=60.0)
            else:
                logger.error("Groq chat completion error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt == max_retries - 1:
                    return ""

    return ""

