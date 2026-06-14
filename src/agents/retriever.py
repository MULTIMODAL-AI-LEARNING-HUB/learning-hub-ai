"""Retriever agent using Qdrant vector search."""

from qdrant_client.models import FieldCondition, Filter, MatchAny
from src.core.clients import get_qdrant_client
from src.utils.embeddings import generate_embedding

COLLECTION_NAME = "document_chunks"


def retrieve(query: str, document_ids: list[str] | None = None, user_id: str | None = None, limit: int = 10) -> list[dict]:
    """Retrieve relevant chunks from Qdrant using vector similarity search."""
    client = get_qdrant_client()
    query_vector = generate_embedding(query)

    query_filter = None
    must_conditions = []
    if document_ids:
        must_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids)))
    if user_id:
        from qdrant_client.models import MatchValue
        must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

    if must_conditions:
        query_filter = Filter(must=must_conditions)

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
