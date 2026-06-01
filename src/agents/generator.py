"""Generator agent using Gemini for high-quality answer generation."""

from src.llm.gemini_client import generate_content

QA_SYSTEM_PROMPT = """Bạn là AI Gia sư thông minh. Dựa vào tài liệu được cung cấp, trả lời câu hỏi của người dùng.

Nguyên tắc:
1. Chỉ sử dụng thông tin từ tài liệu được cung cấp
2. Nếu không có thông tin, nói rõ "Không tìm thấy thông tin trong tài liệu"
3. Trích dẫn nguồn cụ thể (số trang)
4. Trả lời ngắn gọn, dễ hiểu
5. Trả lời bằng cùng ngôn ngữ với câu hỏi"""

SUMMARIZE_SYSTEM_PROMPT = """Bạn là AI tóm tắt tài liệu. Tóm tắt nội dung được cung cấp một cách ngắn gọn và đầy đủ.
Trích dẫn nguồn trang khi có thể."""


def generate_answer(query: str, context_chunks: list[dict], intent: str = "qa") -> dict:
    """Generate answer using Gemini based on retrieved context."""
    if not context_chunks:
        return {
            "answer": "Không tìm thấy thông tin trong tài liệu liên quan đến câu hỏi của bạn.",
            "citations": [],
        }

    context = "\n\n".join(
        [f"[Trang {c.get('page_number', '?')}] {c['text']}" for c in context_chunks]
    )

    system_prompt = SUMMARIZE_SYSTEM_PROMPT if intent == "summarize" else QA_SYSTEM_PROMPT
    user_message = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"

    try:
        answer = generate_content(
            prompt=user_message,
            system_instruction=system_prompt,
        )
    except Exception:
        answer = f"Dựa trên tài liệu, tôi tìm thấy {len(context_chunks)} đoạn liên quan. " + \
                 "\n\n".join([c["text"][:200] for c in context_chunks[:3]])

    citations = [
        {
            "document_id": c.get("document_id", ""),
            "chunk_id": c.get("id", ""),
            "page_number": c.get("page_number"),
            "text": c["text"][:200],
        }
        for c in context_chunks
    ]

    return {"answer": answer, "citations": citations}
