"""Essay grader agent."""

import json

from src.llm.gemini_client import generate_content

ESSAY_GRADER_PROMPT = """Bạn là giáo viên chấm bài.
So sánh bài viết của học sinh với tài liệu gốc để đánh giá.

Đánh giá:
1. Điểm tổng (0-10)
2. Phản hồi chi tiết
3. So sánh các luận điểm với nguồn

Trả về JSON:
{
  "score": float,
  "feedback": "...",
  "comparisons": [{"student_point": "...", "source_match": "...", "similarity": float, "assessment": "..."}]
}"""


def grade_essay(context: str, essay_text: str) -> dict:
    """Grade essay by comparing with source context."""
    prompt = f"""Tài liệu gốc:
{context[:3000]}

Bài viết của học sinh:
{essay_text[:5000]}

Đánh giá và trả về JSON."""

    try:
        response = generate_content(
            prompt=prompt,
            system_instruction=ESSAY_GRADER_PROMPT,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(response)
    except Exception:
        return {
            "score": 5.0,
            "feedback": "Không thể đánh giá tự động. Vui lòng thử lại.",
            "comparisons": [],
        }
