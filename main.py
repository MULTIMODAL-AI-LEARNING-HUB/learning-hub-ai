"""Learning Hub AI Service - Main FastAPI Application."""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.gzip import GZipMiddleware

from src.core.clients import configure_gemini, get_qdrant_client
from src.core.config import settings
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
from src.utils.embeddings import get_embedding_model
from src.workflows.graph import build_graph

workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload embedding model and initialize connection clients on startup."""
    global workflow
    # Preload sentence transformer model
    get_embedding_model()
    # Init Qdrant
    get_qdrant_client()
    # Configure Gemini API
    configure_gemini()
    # Build graph workflow
    workflow = build_graph()
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
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Internal API Key")
    return x_internal_api_key


@app.get("/")
def root():
    return {"message": "Learning Hub AI Service", "status": "running"}


@app.get("/health")
def health_check():
    """Verify health of connection pools and AI models."""
    status_info = {
        "status": "healthy",
        "qdrant": "unknown",
        "embedding_model": "loaded" if get_embedding_model() is not None else "failed",
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
async def chat_ask(payload: QueryRequest, _=Depends(verify_internal_key)) -> ChatResponse:
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
