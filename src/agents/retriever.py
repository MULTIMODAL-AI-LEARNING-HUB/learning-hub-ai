"""Retriever agent using Qdrant vector search."""

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.core.clients import get_qdrant_client
from src.utils.embeddings import generate_embedding

COLLECTION_NAME = "document_chunks"


def retrieve(
    query: str,
    document_ids: list[str] | None = None,
    user_id: str | None = None,
    course_id: str | None = None,
    lesson_id: str | None = None,
    limit: int = 10
) -> list[dict]:
    """Retrieve relevant chunks from Qdrant using vector similarity search.

    Args:
        query: Search query string
        document_ids: Optional list of document IDs to filter by
        user_id: Optional user ID to filter by (for personal documents)
        course_id: Optional course ID to filter by (course-scoped RAG)
        lesson_id: Optional lesson ID to filter by (lesson-scoped RAG)
        limit: Maximum number of results to return

    Returns:
        List of dicts with id, score, and payload from Qdrant
    """
    client = get_qdrant_client()
    query_vector = generate_embedding(query)

    must_conditions = []

    if document_ids:
        must_conditions.append(
            FieldCondition(key="document_id", match=MatchAny(any=document_ids))
        )

    if user_id:
        must_conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        )

    if course_id:
        must_conditions.append(
            FieldCondition(key="course_id", match=MatchValue(value=course_id))
        )

    if lesson_id:
        must_conditions.append(
            FieldCondition(key="lesson_id", match=MatchValue(value=lesson_id))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

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
    except Exception as e:
        print(f"Retriever error: {e}")
        return []


def retrieve_for_course(
    query: str,
    course_id: str,
    lesson_id: str | None = None,
    limit: int = 10
) -> list[dict]:
    """Retrieve relevant chunks from Qdrant for a specific course or lesson.

    This is a convenience function for course-scoped RAG.
    If lesson_id is provided, retrieves only from that lesson.

    Args:
        query: Search query string
        course_id: Course ID to filter by
        lesson_id: Optional lesson ID to filter by (lesson-scoped RAG)
        limit: Maximum number of results to return

    Returns:
        List of dicts with id, score, and payload from Qdrant
    """
    return retrieve(
        query=query,
        course_id=course_id,
        lesson_id=lesson_id,
        limit=limit
    )


def retrieve_with_material_filter(
    query: str,
    material_ids: list[str] | None = None,
    course_id: str | None = None,
    material_type: str | None = None,
    limit: int = 10
) -> list[dict]:
    """Retrieve relevant chunks with additional filtering.

    Args:
        query: Search query string
        material_ids: Optional list of material IDs to filter by
        course_id: Optional course ID to filter by
        material_type: Optional material type (lecture, exercise, etc.)
        limit: Maximum number of results to return

    Returns:
        List of dicts with id, score, and payload from Qdrant
    """
    client = get_qdrant_client()
    query_vector = generate_embedding(query)

    must_conditions = []

    if material_ids:
        must_conditions.append(
            FieldCondition(key="material_id", match=MatchAny(any=material_ids))
        )

    if course_id:
        must_conditions.append(
            FieldCondition(key="course_id", match=MatchValue(value=course_id))
        )

    if material_type:
        must_conditions.append(
            FieldCondition(key="material_type", match=MatchValue(value=material_type))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

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
    except Exception as e:
        print(f"Retriever error: {e}")
        return []