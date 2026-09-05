"""Request schemas for AI service."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=5000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    course_id: str | None = Field(default=None, max_length=100)
    lesson_id: str | None = Field(default=None, max_length=100)


class QuizGenerateRequest(BaseModel):
    context: str = Field(min_length=1, max_length=100_000)
    quiz_type: str = Field(default="quick", pattern="^(quick|detailed)$")
    question_count: int = Field(default=5, ge=1, le=20)


class QuizGenerateFromLessonRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    course_id: str = Field(min_length=1, max_length=100)
    user_id: str | None = Field(default=None, max_length=100)
    question_count: int = Field(default=5, ge=1, le=20)
    lesson_content: str | None = Field(default="", max_length=100_000)



class EssayGradeRequest(BaseModel):
    document_id: str | None = Field(default=None, max_length=100)
    user_id: str | None = Field(default=None, max_length=100)
    context: str | None = Field(default="", max_length=100_000)
    essay_text: str = Field(min_length=1, max_length=50_000)


class FlashcardGenerateRequest(BaseModel):
    context: str = Field(min_length=1, max_length=100_000)
    set_name: str = Field(default="", max_length=255)
    count: int = Field(default=20, ge=1, le=50)
