"""Learning Hub AI Service - Main FastAPI Application."""

import os
import secrets
from contextlib import asynccontextmanager

import json as _json
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

from src.core.clients import configure_gemini, get_qdrant_client
from src.core.config import settings
from src.core.rate_limit import enforce_chat_rate_limit
from src.schemas.requests import (
    EssayGradeRequest,
    FlashcardGenerateRequest,
    QueryRequest,
    QuizGenerateFromLessonRequest,
    QuizGenerateRequest,
)
from src.schemas.responses import (
    ChatResponse,
    Citation,
    EssayGradeResponse,
    FlashcardGenerateResponse,
    FlashcardItem,
    QuizGenerateResponse,
    QuizQuestion,
    TokenUsage,
)
from src.workflows.graph import build_graph

workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize lightweight clients on startup.

    NOTE: the sentence-transformer embedding model (~500MB) is loaded LAZILY
    on first use, never at startup — preloading it OOM-kills small dynos and
    makes /health probes fail before the service can serve traffic.
    Each init step is isolated so one failing dependency (e.g. missing API
    key) never prevents the whole service from booting.
    """
    global workflow
    import logging
    log = logging.getLogger("ai.lifespan")
    try:
        get_qdrant_client()
    except Exception as exc:
        log.warning("Qdrant init deferred: %s", exc)
    try:
        configure_gemini()
    except Exception as exc:
        log.warning("Gemini init deferred: %s", exc)
    try:
        workflow = build_graph()
    except Exception as exc:
        log.warning("Workflow build deferred: %s", exc)
        workflow = None

    # Warm up SentenceTransformer model in a background daemon thread
    # so first user request doesn't suffer 20-40s cold-start latency,
    # while keeping /health and /ready probes fast and unblocked.
    import threading

    def _warmup_embeddings():
        import time
        time.sleep(3)
        try:
            from src.utils.embeddings import get_embedding_model
            get_embedding_model()
            log.info("SentenceTransformer embedding model warmed up successfully")
        except Exception as exc:
            log.warning("Embedding model warmup deferred: %s", exc)

    threading.Thread(target=_warmup_embeddings, daemon=True).start()

    yield


app = FastAPI(
    title="Learning Hub AI Service",
    description="AI/LLM Services for Multimodal Learning Hub",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


async def verify_internal_key(x_internal_api_key: str = Header(..., alias="X-Internal-API-Key")):
    """Verify the shared internal API key for service-to-service communication."""
    if not secrets.compare_digest(x_internal_api_key, settings.INTERNAL_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Internal API Key")
    return x_internal_api_key


@app.get("/")
def root():
    return {"message": "Learning Hub AI Service", "status": "running"}


@app.get("/health")
def health_check():
    """Verify health of connection pools and AI models.

    NOTE: embedding model load is lazy — the /health probe must NOT force a
    ~500MB model download into RAM (would OOM-kill small Heroku dynos).
    """
    status_info = {
        "status": "healthy",
        "qdrant": "unknown",
        "embedding_model": "lazy",
    }

    # Ping Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        status_info["qdrant"] = "healthy"
    except Exception:
        status_info["qdrant"] = "unhealthy"
        status_info["status"] = "degraded"

    return status_info


@app.get("/ready")
def readiness_check():
    return {"ready": True}


@app.post("/chat/ask", response_model=ChatResponse)
async def chat_ask(
    payload: QueryRequest,
    _=Depends(verify_internal_key),
    __=Depends(enforce_chat_rate_limit),
) -> ChatResponse:
    """Process a chat query through the async LangGraph-like workflow.

    Supports both personal documents (document_ids), course-scoped RAG (course_id),
    and lesson-scoped RAG (lesson_id).
    When lesson_id is provided, retrieves only from that lesson.
    When course_id is provided (without lesson_id), retrieves from all course content.
    """
    result = await workflow(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        document_ids=payload.document_ids,
        course_id=payload.course_id,
        lesson_id=payload.lesson_id,
        chat_history=payload.chat_history or [],
    )

    citations = [
        Citation(
            document_id=c.get("document_id", ""),
            chunk_id=c.get("chunk_id", ""),
            page_number=c.get("page_number"),
            text=c.get("text", ""),
            material_id=c.get("material_id", ""),
            course_id=c.get("course_id", ""),
            lesson_id=c.get("lesson_id", ""),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        citations=citations,
        token_usage=TokenUsage(),
    )


@app.post("/chat/ask/stream")
async def chat_ask_stream(
    payload: QueryRequest,
    _=Depends(verify_internal_key),
    __=Depends(enforce_chat_rate_limit),
):
    """Stream chat response via SSE. RAG pipeline runs sync, then Generator streams tokens."""
    import asyncio
    from src.agents.generator import generate_answer_stream
    from src.agents.grader import grade_chunks
    from src.agents.intent import classify_intent
    from src.agents.retriever import retrieve, retrieve_for_course

    async def event_stream():
        try:
            # Step 1: Intent
            try:
                intent_result = await asyncio.wait_for(
                    asyncio.to_thread(classify_intent, payload.query), timeout=10.0
                )
                intent = intent_result.get("intent", "qa")
            except Exception:
                intent = "qa"

            # Step 2: Retrieve (Timeout 60s — matches non-stream workflow;
            # cold embedding model load ~500MB can take 20-50s on first use)
            try:
                if payload.course_id:
                    chunks = await asyncio.wait_for(
                        asyncio.to_thread(
                            retrieve_for_course, payload.query,
                            course_id=payload.course_id, lesson_id=payload.lesson_id, limit=10,
                        ), timeout=60.0,
                    )
                else:
                    chunks = await asyncio.wait_for(
                        asyncio.to_thread(
                            retrieve, payload.query,
                            document_ids=payload.document_ids or None,
                            user_id=payload.user_id or None, limit=10,
                        ), timeout=60.0,
                    )
                    # Soft user_id filter: retry without user_id when the
                    # strict filter yields nothing (mirrors retriever_node).
                    if not chunks and payload.document_ids and payload.user_id:
                        chunks = await asyncio.wait_for(
                            asyncio.to_thread(
                                retrieve, payload.query,
                                document_ids=payload.document_ids or None,
                                user_id=None, limit=10,
                            ), timeout=60.0,
                        )
            except Exception:
                chunks = []

            # Step 3: Grade (citation metadata enrichment happens inside
            # grade_chunks() via enrich_relevant_chunks(), shared with workflow).
            # If the grader times out or returns 0 relevant chunks for explicitly
            # selected documents, fall back to top retrieved chunks so the answer
            # stays grounded in the user's selected material.
            try:
                graded = await asyncio.wait_for(
                    asyncio.to_thread(grade_chunks, payload.query, chunks), timeout=15.0
                )
                relevant_chunks = graded.get("relevant_chunks", [])
            except Exception:
                relevant_chunks = []

            if not relevant_chunks and chunks:
                from src.agents.grader import enrich_relevant_chunks
                raw = [
                    {
                        "id": c.get("id", ""),
                        "text": (c.get("payload", {}) or {}).get("text", ""),
                        "score": c.get("score", 0),
                    }
                    for c in (chunks[:5] if (payload.document_ids or payload.course_id) else chunks)
                ]
                relevant_chunks = enrich_relevant_chunks(raw, chunks)

            citations = [
                {
                    "document_id": c.get("document_id", ""),
                    "chunk_id": c.get("id", ""),
                    "page_number": c.get("page_number"),
                    "text": c.get("text", "")[:200],
                }
                for c in relevant_chunks
            ]
            yield f"data: {_json.dumps({'type': 'meta', 'intent': intent, 'citations': citations})}\n\n"

            # Step 4: Stream Generator
            chat_history = getattr(payload, "chat_history", None) or []
            full_answer = ""
            try:
                async for token in generate_answer_stream(
                    payload.query, relevant_chunks, intent=intent, chat_history=chat_history
                ):
                    full_answer += token
                    yield f"data: {_json.dumps({'type': 'token', 'text': token})}\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            yield f"data: {_json.dumps({'type': 'done', 'answer': full_answer})}\n\n"

        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/study/quiz/generate", response_model=QuizGenerateResponse)
async def generate_quiz(payload: QuizGenerateRequest, _=Depends(verify_internal_key)) -> QuizGenerateResponse:
    """Generate quiz questions from context."""
    from src.agents.quiz import generate_quiz as _generate_quiz

    questions = _generate_quiz(payload.context, payload.quiz_type, payload.question_count)
    return QuizGenerateResponse(
        questions=[
            QuizQuestion(
                id=q["id"],
                question=q["question"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q.get("explanation"),
            )
            for q in questions
        ]
    )


@app.post("/study/quiz/generate-from-lesson", response_model=QuizGenerateResponse)
async def generate_quiz_from_lesson(payload: QuizGenerateFromLessonRequest, _=Depends(verify_internal_key)) -> QuizGenerateResponse:
    """Retrieve lesson material context from Qdrant and generate quiz."""
    from src.agents.quiz import generate_quiz as _generate_quiz
    from src.agents.retriever import retrieve

    # 1. Retrieve chunks from Qdrant associated with the lesson
    chunks = retrieve(query="", lesson_id=payload.lesson_id, limit=20)
    
    # 2. Combine chunk texts
    retrieved_text = "\n".join([chunk["payload"].get("text", "") for chunk in chunks if chunk.get("payload")])
    
    # 3. Combine with direct lesson content if any
    context_parts = []
    if retrieved_text:
        context_parts.append(retrieved_text)
    if payload.lesson_content:
        context_parts.append(payload.lesson_content)
        
    context = "\n\n".join(context_parts)
    if not context:
        # Fallback empty context warning
        context = "No content available. Ask sample general questions."

    # 4. Generate quiz questions using the AI agent
    questions = _generate_quiz(context, "quick", payload.question_count)
    return QuizGenerateResponse(
        questions=[
            QuizQuestion(
                id=q["id"],
                question=q["question"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q.get("explanation"),
            )
            for q in questions
        ]
    )


@app.post("/study/flashcards/generate", response_model=FlashcardGenerateResponse)
async def generate_flashcards(payload: FlashcardGenerateRequest, _=Depends(verify_internal_key)) -> FlashcardGenerateResponse:
    """Generate flashcards from context."""
    from src.agents.flashcard import generate_flashcards as _generate_flashcards

    items = _generate_flashcards(payload.context, payload.set_name, payload.count)
    return FlashcardGenerateResponse(
        items=[
            FlashcardItem(id=item["id"], front=item["front"], back=item["back"])
            for item in items
        ]
    )


@app.post("/study/essay/grade", response_model=EssayGradeResponse)
async def grade_essay(payload: EssayGradeRequest, _=Depends(verify_internal_key)) -> EssayGradeResponse:
    """Grade essay by comparing with source context."""
    from src.agents.essay import grade_essay as _grade_essay
    from src.agents.retriever import retrieve

    context = payload.context or ""
    # Retrieve context from Qdrant if only document_id was provided
    if not context and getattr(payload, "document_id", None):
        results = retrieve(query=payload.essay_text, document_ids=[payload.document_id], user_id=payload.user_id, limit=5)
        context = "\n".join([r["payload"]["text"] for r in results]) if results else ""

    result = _grade_essay(context, payload.essay_text)
    return EssayGradeResponse(
        score=result.get("score", 0),
        feedback=result.get("feedback", ""),
        comparisons=result.get("comparisons", []),
    )


@app.post("/internal/keys/sync")
async def sync_keys(payload: dict, _=Depends(verify_internal_key)):
    """Sync API keys from gateway into in-memory key rotator."""
    from src.llm.key_rotator import GeminiKeyRotator

    keys_list = payload.get("keys", [])
    if not isinstance(keys_list, list) or len(keys_list) > 100:
        raise HTTPException(status_code=400, detail="Invalid key list")
    keys_list = [
        item for item in keys_list
        if isinstance(item, dict)
        and isinstance(item.get("api_key"), str)
        and 16 <= len(item["api_key"].strip()) <= 512
    ]
    GeminiKeyRotator.get_instance().sync_keys(keys_list)
    return {"synced": True, "count": len(keys_list)}


@app.get("/internal/keys/status")
async def get_keys_status(_=Depends(verify_internal_key)):
    """Get active key rotator status and masked key list."""
    from src.llm.key_rotator import GeminiKeyRotator

    return GeminiKeyRotator.get_instance().get_status()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

