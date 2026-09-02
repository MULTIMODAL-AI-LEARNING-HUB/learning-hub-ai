"""Real embedding generation using sentence-transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or load the SentenceTransformer model singleton."""
    from sentence_transformers import SentenceTransformer

    global _model
    if _model is None:
        # Load local lightweight 384-dimensional sentence transformer model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding(text: str, dimension: int = 384) -> list[float]:
    """Generate a real 384-dimensional vector embedding for the given text."""
    model = get_embedding_model()
    embedding = model.encode(text)
    # Ensure it's a standard list of native Python floats
    return [float(v) for v in embedding]
