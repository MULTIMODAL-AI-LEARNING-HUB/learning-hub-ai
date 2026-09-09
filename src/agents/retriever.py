"""Retriever agent using Qdrant vector search."""

import logging

from src.core.clients import get_qdrant_client
from src.utils.embeddings import generate_embedding

logger = logging.getLogger("ai.retriever")

COLLECTION_NAME = "document_chunks"

_INDEXES_ENSURED = False


def _ensure_payload_indexes(client) -> None:
    """Lazily ensure keyword payload indexes exist on filtered fields."""
    global _INDEXES_ENSURED
    if _INDEXES_ENSURED:
        return
    try:
        from qdrant_client.models import PayloadSchemaType
        for field in ("document_id", "user_id", "course_id", "lesson_id", "material_id"):
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        _INDEXES_ENSURED = True
    except Exception as exc:
        logger.warning("Could not ensure payload indexes: %s", exc)


def _build_filter(
    *,
    document_ids: list[str] | None = None,
    user_id: str | None = None,
    course_id: str | None = None,
    lesson_id: str | None = None,
    material_ids: list[str] | None = None,
    material_type: str | None = None,
):
    """Build a Qdrant filter without importing Qdrant during test collection."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must_conditions = []

    if document_ids:
        if len(document_ids) == 1:
            must_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))
            )
        else:
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
    if material_ids:
        if len(material_ids) == 1:
            must_conditions.append(
                FieldCondition(key="material_id", match=MatchValue(value=material_ids[0]))
            )
        else:
            must_conditions.append(
                FieldCondition(key="material_id", match=MatchAny(any=material_ids))
            )
    if material_type:
        must_conditions.append(
            FieldCondition(key="material_type", match=MatchValue(value=material_type))
        )

    return Filter(must=must_conditions) if must_conditions else None


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
        course_id: Course ID to filter by (course-scoped RAG)
        lesson_id: Optional lesson ID to filter by (lesson-scoped RAG)
        limit: Maximum number of results to return

    Returns:
        List of dicts with id, score, and payload from Qdrant
    """
    client = get_qdrant_client()
    _ensure_payload_indexes(client)
    query_vector = generate_embedding(query)

    query_filter = _build_filter(
        document_ids=document_ids,
        user_id=user_id,
        course_id=course_id,
        lesson_id=lesson_id,
    )

    try:
        points = None
        if hasattr(client, "search"):
            points = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
        elif hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            points = res.points if hasattr(res, "points") else res

        if points is None:
            logger.warning("Retriever: neither search nor query_points available on client")
            return []

        hits = [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in points
        ]
        logger.info(
            "Retriever query='%s' doc_ids=%s user_id=%s hits=%d",
            query[:50],
            document_ids,
            user_id,
            len(hits),
        )
        return hits
    except Exception as e:
        logger.error("Retriever error: %s (doc_ids=%s user_id=%s)", e, document_ids, user_id, exc_info=True)
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
    _ensure_payload_indexes(client)
    query_vector = generate_embedding(query)

    query_filter = _build_filter(
        material_ids=material_ids,
        course_id=course_id,
        material_type=material_type,
    )

    try:
        points = None
        if hasattr(client, "search"):
            points = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
        elif hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            points = res.points if hasattr(res, "points") else res

        if points is None:
            logger.warning("Retriever(material): no search method on client")
            return []

        hits = [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in points
        ]
        logger.info(
            "Retriever(material) query='%s' hits=%d",
            query[:50],
            len(hits),
        )
        return hits
    except Exception as e:
        logger.error("Retriever(material) error: %s", e, exc_info=True)
        return []
