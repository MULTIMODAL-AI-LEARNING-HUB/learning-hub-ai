"""Third-party client singletons for AI service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.core.config import settings

if TYPE_CHECKING:
    from groq import Groq
    from qdrant_client import QdrantClient

_qdrant_client: Optional[QdrantClient] = None
_groq_client: Optional[Groq] = None
_gemini_configured: bool = False


def get_qdrant_client() -> QdrantClient:
    """Retrieve the QdrantClient singleton."""
    from qdrant_client import QdrantClient

    global _qdrant_client
    if _qdrant_client is None:
        if settings.QDRANT_URL:
            _qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            _qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
    return _qdrant_client


def get_groq_client() -> Groq:
    """Retrieve the Groq client singleton."""
    from groq import Groq

    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def configure_gemini() -> None:
    """Initialize and configure Gemini API."""
    import google.generativeai as genai

    global _gemini_configured
    if not _gemini_configured:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _gemini_configured = True
