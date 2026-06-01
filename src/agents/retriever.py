"""Retriever agent using Qdrant vector search."""

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from src.core.config import settings
from src.utils.embeddings import generate_embedding

COLLECTION_NAME = "document_chunks"

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _client


def retrieve(query: str, document_ids: list[str] | None = None, limit: int = 10) -> list[dict]:
    """Retrieve relevant chunks from Qdrant using vector similarity search."""
    client = get_qdrant()
    query_vector = generate_embedding(query)

    query_filter = None
    if document_ids:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))]
        )

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results.points
        ]
    except Exception:
        return []
