"""Gemini client wrapper for high-quality generation."""

import google.generativeai as genai

from src.core.config import settings

_configured = False


def _configure():
    global _configured
    if not _configured:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _configured = True


def generate_content(prompt: str, system_instruction: str | None = None) -> str:
    _configure()
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system_instruction,
    )
    response = model.generate_content(prompt)
    return response.text or ""


def chat(messages: list[dict], system_instruction: str | None = None) -> str:
    _configure()
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system_instruction,
    )
    chat_session = model.start_chat(history=messages[:-1] if messages else [])
    last_msg = messages[-1]["content"] if messages else ""
    response = chat_session.send_message(last_msg)
    return response.text or ""
