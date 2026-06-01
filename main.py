"""Learning Hub AI Service - Main FastAPI Application."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


@asynccontextmanager
async def lifespan(app: FastAPI):
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

workflow = build_graph()


@app.get("/")
def root():
    return {"message": "Learning Hub AI Service", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "learning-hub-ai"}


@app.get("/ready")
def readiness_check():
    return {"ready": True}


@app.post("/chat/ask", response_model=ChatResponse)
async def chat_ask(payload: QueryRequest) -> ChatResponse:
    """Process a chat query through the LangGraph workflow."""
    result = workflow(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        document_ids=payload.document_ids,
    )

    citations = [
        Citation(
            document_id=c.get("document_id", ""),
            chunk_id=c.get("chunk_id", ""),
            page_number=c.get("page_number"),
            text=c.get("text", ""),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        citations=citations,
        token_usage=TokenUsage(),
    )


@app.post("/study/quiz/generate", response_model=QuizGenerateResponse)
async def generate_quiz(payload: QuizGenerateRequest) -> QuizGenerateResponse:
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
async def generate_flashcards(payload: FlashcardGenerateRequest) -> FlashcardGenerateResponse:
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
async def grade_essay(payload: EssayGradeRequest) -> EssayGradeResponse:
    """Grade essay by comparing with source context."""
    from src.agents.essay import grade_essay as _grade_essay

    result = _grade_essay(payload.context, payload.essay_text)
    return EssayGradeResponse(
        score=result.get("score", 0),
        feedback=result.get("feedback", ""),
        comparisons=result.get("comparisons", []),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
