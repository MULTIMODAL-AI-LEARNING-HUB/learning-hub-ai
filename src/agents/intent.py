"""Intent classifier agent using Groq (fast, low cost)."""

import json
from src.llm.groq_client import chat_completion

INTENT_SYSTEM_PROMPT = """Bạn là một classifier phân loại ý định người dùng.
Phân loại câu hỏi vào một trong các loại:
- qa: Hỏi đáp thông thường
- quiz: Yêu cầu tạo câu hỏi trắc nghiệm
- flashcard: Yêu cầu tạo flashcard
- essay_grading: Yêu cầu chấm bài tự luận
- summarize: Yêu cầu tóm tắt nội dung
- other: Khác

Chỉ trả về JSON: {"intent": "...", "sub_intent": "..."}
Không giải thích thêm."""


def classify_intent(query: str) -> dict:
    """Classify user intent using Groq LLM."""
    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(response)
    except Exception:
        return {"intent": "qa", "sub_intent": "default"}
