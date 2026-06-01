"""Response schemas for AI service."""

from pydantic import BaseModel


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    page_number: int | None = None
    text: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    token_usage: TokenUsage = TokenUsage()


class QuizQuestion(BaseModel):
    id: str
    question: str
    options: list[str]
    correct_answer: str


class QuizGenerateResponse(BaseModel):
    questions: list[QuizQuestion]


class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str


class FlashcardGenerateResponse(BaseModel):
    items: list[FlashcardItem]


class EssayGradeResponse(BaseModel):
    score: float
    feedback: str
    comparisons: list[dict] = []
