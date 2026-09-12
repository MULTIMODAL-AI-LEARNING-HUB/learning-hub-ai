"""Remedial Quiz Generator Agent.

Specialized in diagnosing student knowledge gaps from incorrect answers
and generating targeted micro-quizzes (scaffolded questions) to reinforce weak concepts.
"""

import json
import logging
import uuid

from src.agents.quiz import _looks_placeholder, _normalize_options_and_answer
from src.llm.gemini_client import generate_content
from src.utils.prompt_safety import untrusted_text

logger = logging.getLogger(__name__)

REMEDIAL_SYSTEM_PROMPT = """Bạn là chuyên gia sư phạm chẩn đoán và khắc phục lỗ hổng kiến thức chuẩn cá nhân hóa.
Nhiệm vụ: Dựa vào nội dung bài học và danh sách các câu hỏi mà học viên ĐÃ LÀM SAI, hãy thiết kế một bài trắc nghiệm củng cố (Micro-Quiz) nhằm giúp học viên nắm vững lại các khái niệm chưa hiểu rõ.

Nguyên tắc biên soạn:
1. Tập trung vào lỗ hổng: Các câu hỏi mới phải xoay quanh ĐÚNG các khái niệm, cơ chế, định nghĩa trong các câu học viên đã làm sai.
2. KHÔNG lặp lại nguyên văn câu hỏi cũ: Hãy thay đổi ngữ cảnh, đặt tình huống thực tế, hoặc hỏi từ góc nhìn ngược lại để kiểm tra độ hiểu sâu (chứ không phải học vẹt).
3. Bốn phương án lựa chọn (A, B, C, D): Rõ ràng, phương án đúng chuẩn xác, các phương án nhiễu phản ánh các hiểu lầm phổ biến.
4. Lời giải thích (explanation): BẮT BUỘC chỉ rõ TẠI SAO đáp án này đúng, đồng thời phân tích lỗi tư duy/hiểu sai mà học viên dễ mắc phải.

Định dạng trả về: DUY NHẤT một JSON array hợp lệ, không bọc văn bản phụ:
[
  {
    "question": "Trong kịch bản X, tại sao cơ chế Y lại...",
    "options": ["Phương án 1", "Phương án 2", "Phương án 3", "Phương án 4"],
    "correct_answer": "B",
    "explanation": "Căn cứ bài học: Y hoạt động theo nguyên lý... Lỗi phổ biến là nhầm lẫn với Z."
  }
]
"""


def generate_remedial_quiz(
    missed_questions: list[dict],
    lesson_content: str = "",
    lesson_title: str = "",
    question_count: int = 3,
) -> list[dict]:
    """Generate targeted remedial questions addressing specific missed concepts."""
    # Format missed questions cleanly for the LLM
    missed_summary = []
    for idx, q in enumerate(missed_questions[:10], 1):
        q_text = q.get("question_text") or q.get("question") or "Câu hỏi không xác định"
        selected = q.get("selected_answers") or q.get("your_answer") or "Chưa trả lời"
        correct = q.get("correct_answers") or q.get("correct_answer") or "Không có"
        explanation = q.get("explanation") or ""
        missed_summary.append(
            f"Lỗ hổng {idx}:\n"
            f"- Câu đã làm sai: {q_text}\n"
            f"- Học viên đã chọn sai: {selected}\n"
            f"- Đáp án đúng cần nắm: {correct}\n"
            f"- Căn cứ/Giải thích: {explanation}"
        )

    missed_text = "\n\n".join(missed_summary)
    safe_lesson = untrusted_text("LESSON_CONTENT", lesson_content or f"Chủ đề bài học: {lesson_title}", 15_000)

    prompt = f"""Học viên vừa làm bài trắc nghiệm về bài học: "{lesson_title or 'Bài học'}" và đã làm SAI các câu sau:

{missed_text}

NỘI DUNG THAM CHIẾU BÀI HỌC:
{safe_lesson}

Hãy biên soạn đúng {question_count} câu hỏi trắc nghiệm mới bám sát các lỗ hổng kiến thức trên để giúp học viên củng cố lại.
Yêu cầu:
- Mỗi câu hỏi gồm đúng 4 lựa chọn (A, B, C, D).
- Có explanation chi tiết phân tích lỗi sai và cách hiểu đúng.
- Trả về DUY NHẤT một JSON array hợp lệ.
"""

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = generate_content(
                prompt=prompt,
                system_instruction=REMEDIAL_SYSTEM_PROMPT,
            )
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            questions = json.loads(response)
            if isinstance(questions, dict) and "questions" in questions:
                questions = questions["questions"]
            if not isinstance(questions, list):
                raise ValueError("Expected JSON array of remedial questions")

            normalized = [_normalize_options_and_answer(q) for q in questions if isinstance(q, dict)]
            normalized = [q for q in normalized if q.get("question") and not _looks_placeholder(q)]

            if normalized:
                return normalized[:question_count]
            raise ValueError("No valid questions generated after normalization")
        except Exception as exc:
            last_error = exc
            logger.warning("Remedial quiz generation attempt %d failed: %s", attempt + 1, exc)
            continue

    logger.error("All remedial quiz attempts failed: %s", last_error)
    # Return minimal fallback grounded question if all LLM retries fail
    fallback_q = {
        "id": str(uuid.uuid4()),
        "question": f"Khái niệm trọng tâm cần ôn tập lại trong bài '{lesson_title or 'bài học'}':",
        "options": [
            "Đọc kỹ lại phần lý thuyết và ví dụ minh họa",
            "Bỏ qua và chuyển sang bài học tiếp theo",
            "Chỉ học thuộc đáp án mà không hiểu nguyên lý",
            "Không cần làm lại bài tập",
        ],
        "correct_answer": "A",
        "explanation": "Khi gặp lỗ hổng kiến thức, xem lại lý thuyết và làm lại bài kiểm tra giúp ghi nhớ sâu sắc hơn.",
    }
    return [fallback_q]
