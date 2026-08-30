"""Reflection agent for self-checking answer quality."""

import json

from src.llm.groq_client import chat_completion

REFLECTION_SYSTEM_PROMPT = """Bạn là một AI kiểm tra chất lượng câu trả lời.
Kiểm tra các điểm sau:
1. Câu trả lời có dựa trên context không?
2. Có hallucination không?
3. Citations có chính xác không?
4. Câu trả lời có đầy đủ không?

Trả về JSON:
{
  "needs_reflection": true/false,
  "feedback": "...",
  "issues": ["issue1", "issue2"]
}"""


def reflect(answer: str, context_chunks: list[dict], query: str) -> dict:
    """Self-check answer quality."""
    context_text = "\n".join([c["text"][:200] for c in context_chunks[:3]])

    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Câu hỏi: {query}\n\nContext:\n{context_text}\n\nCâu trả lời: {answer}",
                },
            ],
            temperature=0.1,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(response)
    except Exception:
        return {"needs_reflection": False, "feedback": "", "issues": []}
