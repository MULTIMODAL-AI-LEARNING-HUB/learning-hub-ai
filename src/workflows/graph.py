"""LangGraph workflow for multi-agent AI system."""

import asyncio
import logging
import time
from typing import TypedDict

from src.agents.generator import generate_answer
from src.agents.grader import grade_chunks
from src.agents.intent import classify_intent
from src.agents.reflection import reflect
from src.agents.retriever import retrieve, retrieve_for_course

logger = logging.getLogger("ai.workflow")


class GraphState(TypedDict):
    query: str
    session_id: str
    user_id: str
    course_id: str | None
    lesson_id: str | None
    tutor_mode: str
    document_ids: list[str]
    chat_history: list[dict]
    intent: str
    sub_intent: str
    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    current_answer: str
    citations: list[dict]
    needs_reflection: bool
    reflection_feedback: str
    final_answer: str


def intent_node(state: GraphState) -> GraphState:
    """Classify user intent."""
    result = classify_intent(state["query"])
    state["intent"] = result.get("intent", "qa")
    state["sub_intent"] = result.get("sub_intent", "default")
    return state


def retriever_node(state: GraphState) -> GraphState:
    """Retrieve relevant chunks from Qdrant.

    Supports both personal documents and course-scoped retrieval.
    If course_id is provided, retrieves only from that course.
    If lesson_id is also provided, retrieves only from that lesson.
    Otherwise uses document_ids/user_id for filtering.

    NOTE: user_id filter is applied as a soft preference, not a hard
    requirement. Worker-upserted chunks may carry user_id as string UUID
    while the gateway forwards user_id in a different string form, and
    personal-doc chunks may lack user_id entirely. A strict AND filter
    would then return zero hits even when document_ids match. So: first
    try the strict filter, and if it yields nothing, retry with
    document_ids only (dropping user_id) before giving up.
    """
    course_id = state.get("course_id")
    lesson_id = state.get("lesson_id")
    doc_ids = state.get("document_ids") or None
    user_id = state.get("user_id") or None

    if course_id:
        chunks = retrieve_for_course(
            state["query"],
            course_id=course_id,
            lesson_id=lesson_id,
            limit=10
        )
    else:
        chunks = retrieve(
            state["query"],
            document_ids=doc_ids,
            user_id=user_id,
            limit=10
        )
        if not chunks and doc_ids and user_id:
            logger.info(
                "node=retriever strict filter empty (doc_ids=%d), "
                "retrying without user_id session=%s",
                len(doc_ids), state.get("session_id", ""),
            )
            chunks = retrieve(
                state["query"],
                document_ids=doc_ids,
                user_id=None,
                limit=10
            )

    state["retrieved_chunks"] = chunks
    return state


def grader_node(state: GraphState) -> GraphState:
    """Grade relevance of retrieved chunks.

    Citation metadata enrichment (document_id/page_number/...) is handled
    inside grade_chunks() via enrich_relevant_chunks(), shared with the
    /chat/ask/stream endpoint.
    """
    result = grade_chunks(state["query"], state["retrieved_chunks"])
    state["relevant_chunks"] = result.get("relevant_chunks", [])
    return state


def generator_node(state: GraphState) -> GraphState:
    """Generate answer based on relevant chunks."""
    result = generate_answer(
        state["query"],
        state["relevant_chunks"],
        intent=state["intent"],
        chat_history=state.get("chat_history") or [],
        tutor_mode=state.get("tutor_mode", "standard"),
    )
    state["current_answer"] = result.get("answer", "")
    state["citations"] = result.get("citations", [])
    return state


def reflection_node(state: GraphState) -> GraphState:
    """Self-check answer quality."""
    result = reflect(state["current_answer"], state["relevant_chunks"], state["query"])
    state["needs_reflection"] = result.get("needs_reflection", False)
    state["reflection_feedback"] = result.get("feedback", "")
    return state


def finalize_node(state: GraphState) -> GraphState:
    """Finalize the answer."""
    state["final_answer"] = state["current_answer"]
    return state


def should_retry(state: GraphState) -> str:
    """Decide whether to retry generation."""
    if state.get("needs_reflection") and state.get("reflection_feedback"):
        return "retry"
    return "end"


def build_graph():
    """Build an asynchronous workflow execution engine.

    Supports course-scoped RAG when course_id is provided.
    When course_id is None, uses personal documents (document_ids/user_id).
    """
    async def run_workflow(
        query: str,
        session_id: str,
        user_id: str,
        document_ids: list[str] | None = None,
        course_id: str | None = None,
        lesson_id: str | None = None,
        chat_history: list[dict] | None = None,
        tutor_mode: str = "standard",
    ) -> dict:
        state: GraphState = {
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "course_id": course_id,
            "lesson_id": lesson_id,
            "tutor_mode": tutor_mode or "standard",
            "document_ids": document_ids or [],
            "chat_history": chat_history or [],
            "intent": "",
            "sub_intent": "",
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "current_answer": "",
            "citations": [],
            "needs_reflection": False,
            "reflection_feedback": "",
            "final_answer": "",
        }

        # Step 1: Intent classification (Timeout 10s)
        _t0 = time.monotonic()
        try:
            state = await asyncio.wait_for(asyncio.to_thread(intent_node, state), timeout=10.0)
        except (asyncio.TimeoutError, Exception):
            state["intent"] = "qa"
            state["sub_intent"] = "default"
        logger.info("node=intent latency_ms=%.0f session=%s intent=%s", (time.monotonic() - _t0) * 1000, state.get("session_id", ""), state.get("intent", ""))

        # Route based on intent. Greetings (hi/hello/...) skip RAG entirely:
        # short-circuit with a friendly reply and empty citations instead of
        # force-attaching 10 irrelevant chunks and "answering from documents".
        if state["intent"] == "greeting":
            from src.agents.generator import generate_greeting_reply
            state["current_answer"] = generate_greeting_reply(state["query"], tutor_mode=state.get("tutor_mode", "standard"))
            state["citations"] = []
            state = finalize_node(state)
        elif state["intent"] in ("qa", "summarize"):
            # Step 2: Retrieve (Timeout 60s — cold embedding model load
            # ~500MB sentence-transformers can take 20-50s on first use)
            _t0 = time.monotonic()
            try:
                state = await asyncio.wait_for(asyncio.to_thread(retriever_node, state), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("node=retriever timeout session=%s", state.get("session_id", ""))
                state["retrieved_chunks"] = []
            except Exception:
                state["retrieved_chunks"] = []
            logger.info("node=retriever latency_ms=%.0f session=%s chunks=%d", (time.monotonic() - _t0) * 1000, state.get("session_id", ""), len(state.get("retrieved_chunks", [])))

            # Step 3: Grade (Timeout 15s). On timeout/LLM failure, fall back to
            # the raw retrieved chunks (enriched with citation metadata) instead
            # of wiping relevant_chunks — otherwise the generator answers
            # ungrounded and citations come back empty.
            _t0 = time.monotonic()
            try:
                state = await asyncio.wait_for(asyncio.to_thread(grader_node, state), timeout=15.0)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(
                    "node=grader fallback to raw chunks session=%s reason=%s retrieved=%d",
                    state.get("session_id", ""), type(exc).__name__,
                    len(state.get("retrieved_chunks", [])),
                )
                try:
                    from src.agents.grader import enrich_relevant_chunks
                    raw = [
                        {
                            "id": c.get("id", ""),
                            "text": (c.get("payload", {}) or {}).get("text", ""),
                            "score": c.get("score", 0),
                        }
                        for c in state.get("retrieved_chunks", [])
                    ]
                    state["relevant_chunks"] = enrich_relevant_chunks(raw, state.get("retrieved_chunks", []))
                except Exception:
                    state["relevant_chunks"] = []
            logger.info("node=grader latency_ms=%.0f session=%s relevant=%d", (time.monotonic() - _t0) * 1000, state.get("session_id", ""), len(state.get("relevant_chunks", [])))

            # If grader returned 0 relevant chunks but we had chunks from user-selected
            # documents (or course), fall back to top retrieved chunks so the answer
            # remains grounded in the user's selected material rather than hallucinating.
            if not state.get("relevant_chunks") and state.get("retrieved_chunks") and (state.get("document_ids") or state.get("course_id")):
                logger.info(
                    "node=grader empty relevance for explicit docs session=%s falling back to top %d retrieved chunks",
                    state.get("session_id", ""), min(5, len(state["retrieved_chunks"])),
                )
                from src.agents.grader import enrich_relevant_chunks
                raw = [
                    {
                        "id": c.get("id", ""),
                        "text": (c.get("payload", {}) or {}).get("text", ""),
                        "score": c.get("score", 0),
                    }
                    for c in state["retrieved_chunks"][:5]
                ]
                state["relevant_chunks"] = enrich_relevant_chunks(raw, state["retrieved_chunks"])

            # Step 4: Generate (Timeout 60s)
            _t0 = time.monotonic()
            try:
                state = await asyncio.wait_for(asyncio.to_thread(generator_node, state), timeout=60.0)
            except (asyncio.TimeoutError, Exception):
                state["current_answer"] = "AI generation timed out. Please try again."
                state["citations"] = []
            logger.info("node=generator latency_ms=%.0f session=%s answer_len=%d", (time.monotonic() - _t0) * 1000, state.get("session_id", ""), len(state.get("current_answer", "")))

            # Step 5: Reflect (Timeout 15s, skipped after grader failure).
            # When the grader timed out (relevant_chunks empty), self-checking
            # an ungrounded fallback answer and re-running the generator adds
            # up to 75s of pure latency — skip it and finalize immediately.
            _had_relevant = bool(state.get("relevant_chunks"))
            if _had_relevant:
                _t0 = time.monotonic()
                try:
                    state = await asyncio.wait_for(asyncio.to_thread(reflection_node, state), timeout=15.0)
                except (asyncio.TimeoutError, Exception):
                    state["needs_reflection"] = False
                logger.info("node=reflector latency_ms=%.0f session=%s needs_retry=%s", (time.monotonic() - _t0) * 1000, state.get("session_id", ""), state.get("needs_reflection", False))

                # Step 6: Retry if needed
                if should_retry(state) == "retry":
                    try:
                        state = await asyncio.wait_for(asyncio.to_thread(generator_node, state), timeout=60.0)
                    except (asyncio.TimeoutError, Exception):
                        pass
            else:
                logger.info("node=reflector skipped session=%s reason=no_relevant_chunks", state.get("session_id", ""))
                state["needs_reflection"] = False

            state = finalize_node(state)
        else:
            # For non-QA intents, just generate directly (Timeout 60s)
            try:
                state = await asyncio.wait_for(asyncio.to_thread(generator_node, state), timeout=60.0)
            except (asyncio.TimeoutError, Exception):
                state["current_answer"] = "AI generation timed out. Please try again."
            state = finalize_node(state)

        return {
            "answer": state["final_answer"] or state["current_answer"] or "I'm sorry, I was unable to generate a response. Please try again.",
            "citations": state["citations"],
            "intent": state["intent"],
        }

    return run_workflow