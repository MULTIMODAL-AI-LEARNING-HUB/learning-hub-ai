"""Learning Hub AI Service - Main FastAPI Application."""

import os
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from src.schemas.requests import QueryRequest, QuizGenerateRequest, EssayGradeRequest, FlashcardGenerateRequest
from src.schemas.responses import (
    ChatResponse,
    Citation,
    TokenUsage,
    QuizGenerateResponse,
    QuizQuestion,
    FlashcardGenerateResponse,
    FlashcardItem,
    EssayGradeResponse,
)
from src.workflows.graph import build_graph
from src.core.config import settings
from src.core.clients import get_qdrant_client, configure_gemini
from src.utils.embeddings import get_embedding_model


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    Supports both personal documents (document_ids) and course-scoped RAG (course_id).
    When course_id is provided, retrieves only from that course's materials.
    """
    result = await workflow(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        document_ids=payload.document_ids,
        course_id=payload.course_id,
    )

    citations = [
        Citation(
            document_id=c.get("document_id", ""),
            chunk_id=c.get("chunk_id", ""),
            page_number=c.get("page_number"),
            text=c.get("text", ""),
            material_id=c.get("material_id", ""),
            course_id=c.get("course_id", ""),
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
