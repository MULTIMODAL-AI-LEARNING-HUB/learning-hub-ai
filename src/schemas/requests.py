"""Request schemas for AI service."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    session_id: str
    user_id: str
    query: str
    document_ids: list[str] = []
    course_id: str | None = None
    lesson_id: str | None = None


class QuizGenerateRequest(BaseModel):
    context: str
    quiz_type: str = "quick"
    question_count: int = 5


class EssayGradeRequest(BaseModel):
    document_id: str | None = None
    user_id: str | None = None
    context: str | None = ""
    essay_text: str


class FlashcardGenerateRequest(BaseModel):
    context: str
    set_name: str = ""
    count: int = 20