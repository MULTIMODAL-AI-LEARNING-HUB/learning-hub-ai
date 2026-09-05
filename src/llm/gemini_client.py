"""Gemini client wrapper with Key Rotation and Auto-Failover."""

import logging
from src.core.config import settings
from src.llm.key_rotator import GeminiKeyRotator

logger = logging.getLogger("ai.gemini_client")


def _is_rate_limit_or_quota_error(e: Exception) -> bool:
    err_str = str(e).lower()
    err_type = type(e).__name__.lower()
    return any(term in err_str or term in err_type for term in ["429", "resourceexhausted", "quota", "rate limit", "too many requests"])


def generate_content(prompt: str, system_instruction: str | None = None) -> str:
    import google.generativeai as genai

    rotator = GeminiKeyRotator.get_instance()
    max_attempts = max(1, min(len(rotator.keys), 5)) if rotator.keys else 1
    last_error = None

    for attempt in range(max_attempts):
        key = rotator.get_next_key()
        genai.configure(api_key=key)
        try:
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=system_instruction,
            )
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as e:
            last_error = e
            if _is_rate_limit_or_quota_error(e):
                rotator.report_rate_limit(key, cooldown_seconds=60.0)
                logger.warning("Gemini key rate-limited on attempt %d/%d. Retrying with next key...", attempt + 1, max_attempts)
                continue
            logger.error("Gemini generate_content error: %s", e)
            raise e

    if last_error:
        raise last_error
    return ""


def chat(messages: list[dict], system_instruction: str | None = None) -> str:
    import google.generativeai as genai

    rotator = GeminiKeyRotator.get_instance()
    max_attempts = max(1, min(len(rotator.keys), 5)) if rotator.keys else 1
    last_error = None

    for attempt in range(max_attempts):
        key = rotator.get_next_key()
        genai.configure(api_key=key)
        try:
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=system_instruction,
            )
            chat_session = model.start_chat(history=messages[:-1] if messages else [])
            last_msg = messages[-1]["content"] if messages else ""
            response = chat_session.send_message(last_msg)
            return response.text or ""
        except Exception as e:
            last_error = e
            if _is_rate_limit_or_quota_error(e):
                rotator.report_rate_limit(key, cooldown_seconds=60.0)
                logger.warning("Gemini chat key rate-limited on attempt %d/%d. Retrying with next key...", attempt + 1, max_attempts)
                continue
            logger.error("Gemini chat error: %s", e)
            raise e

    if last_error:
        raise last_error
    return ""

