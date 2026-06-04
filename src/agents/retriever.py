"""Retriever agent using Qdrant vector search."""

from qdrant_client.models import FieldCondition, Filter, MatchAny
from src.core.clients import get_qdrant_client
from src.utils.embeddings import generate_embedding

COLLECTION_NAME = "document_chunks"


def retrieve(query: str, document_ids: list[str] | None = None, limit: int = 10) -> list[dict]:
    """Retrieve relevant chunks from Qdrant using vector similarity search."""
    client = get_qdrant_client()
    query_vector = generate_embedding(query)

    query_filter = None
    if document_ids:
        # Support searching across multiple document IDs using MatchAny
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))]
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
