"""Quiz generator agent."""

import json
from src.llm.gemini_client import generate_content
import uuid

QUIZ_SYSTEM_PROMPT = """Bạn là một AI tạo câu hỏi trắc nghiệm.
Dựa vào tài liệu, tạo câu hỏi trắc nghiệm với 4 lựa chọn A, B, C, D.
Trả về JSON array:
[{"question": "...", "options": ["A", "B", "C", "D"], "correct_answer": "A"}]
Đảm bảo JSON hợp lệ."""


def generate_quiz(context: str, quiz_type: str = "quick", question_count: int = 5) -> list[dict]:
    """Generate quiz questions from context."""
    prompt = f"""Dựa vào tài liệu sau, tạo {question_count} câu hỏi trắc nghiệm.
Loại quiz: {quiz_type}

Tài liệu:
{context[:3000]}

Trả về JSON array."""

    try:
        response = generate_content(
            prompt=prompt,
            system_instruction=QUIZ_SYSTEM_PROMPT,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        questions = json.loads(response)
        if isinstance(questions, dict) and "questions" in questions:
            questions = questions["questions"]
        return [
            {
                "id": str(uuid.uuid4()),
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "correct_answer": q.get("correct_answer", "A"),
            }
            for q in questions[:question_count]
        ]
    except Exception:
        return [
            {
                "id": str(uuid.uuid4()),
                "question": f"Câu hỏi mẫu {i+1}: Dựa trên nội dung tài liệu?",
                "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
                "correct_answer": "A",
            }
            for i in range(min(question_count, 5))
        ]
