"""Flashcard generator agent."""

import json
import uuid
from src.llm.gemini_client import generate_content

FLASHCARD_PROMPT = """Bạn là AI tạo flashcard học tập.
Dựa vào tài liệu, tạo flashcard với mặt trước (câu hỏi) và mặt sau (trả lời).
Trả về JSON array:
[{"front": "Câu hỏi", "back": "Trả lời"}]
Trả về JSON hợp lệ."""


def generate_flashcards(context: str, set_name: str = "", count: int = 20) -> list[dict]:
    """Generate flashcards from context."""
    prompt = f"""Dựa vào tài liệu sau, tạo {count} flashcard.

Tài liệu:
{context[:3000]}

Trả về JSON array."""

    try:
        response = generate_content(
            prompt=prompt,
            system_instruction=FLASHCARD_PROMPT,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        items = json.loads(response)
        if isinstance(items, dict) and "items" in items:
            items = items["items"]
        return [
            {
                "id": str(uuid.uuid4()),
                "front": item.get("front", item.get("question", "")),
                "back": item.get("back", item.get("answer", "")),
            }
            for item in items[:count]
        ]
    except Exception:
        return [
            {
                "id": str(uuid.uuid4()),
                "front": f"Flashcard {i+1}: Mẫu câu hỏi từ tài liệu",
                "back": f"Flashcard {i+1}: Mẫu câu trả lời từ tài liệu",
            }
            for i in range(min(count, 10))
        ]
