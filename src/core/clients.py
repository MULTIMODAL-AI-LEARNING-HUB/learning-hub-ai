"""Third-party clients singletons for AI service."""

from typing import Optional
from qdrant_client import QdrantClient
from groq import Groq
import google.generativeai as genai
from src.core.config import settings

_qdrant_client: Optional[QdrantClient] = None
_groq_client: Optional[Groq] = None
_gemini_configured: bool = False


def get_qdrant_client() -> QdrantClient:
    """Retrieve the QdrantClient singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def get_groq_client() -> Groq:
    """Retrieve the Groq client singleton."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def configure_gemini() -> None:
    """Initialize and configure Gemini API."""
    global _gemini_configured
    if not _gemini_configured:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _gemini_configured = True
