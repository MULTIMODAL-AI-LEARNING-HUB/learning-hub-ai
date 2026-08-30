"""Grader agent - evaluates relevance of retrieved chunks."""

import json

from src.llm.groq_client import chat_completion

GRADER_SYSTEM_PROMPT = """Bạn là một grader đánh giá mức độ liên quan của document chunks.
Cho mỗi chunk, đánh giá từ 0-1:
- 1: Hoàn toàn liên quan đến câu hỏi
- 0.5: Liên quan một phần
- 0: Không liên quan

Chỉ giữ lại các chunk có score >= 0.5.
Trả về JSON: {"relevant_chunks": [{"id": "...", "text": "...", "score": float, "page_number": int}], "avg_score": float}"""


def grade_chunks(query: str, chunks: list[dict]) -> dict:
    """Grade relevance of retrieved chunks."""
    if not chunks:
        return {"relevant_chunks": [], "avg_score": 0}

    chunks_text = "\n\n".join(
        [f"[{i}] (page {c['payload'].get('page_number', '?')}) {c['payload']['text'][:300]}" for i, c in enumerate(chunks)]
    )

    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Câu hỏi: {query}\n\nChunks:\n{chunks_text}"},
            ],
            temperature=0.1,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(response)
    except Exception:
        return {
            "relevant_chunks": [
                {
                    "id": c["id"],
                    "text": c["payload"]["text"],
                    "score": c["score"],
                    "page_number": c["payload"].get("page_number"),
                }
                for c in chunks
            ],
            "avg_score": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0,
        }
