"""Interactive Knowledge Mindmap Generator Agent.

Generates hierarchical Markdown trees optimized for Markmap SVG/D3 visualization.
"""

import logging
import re

from src.llm.gemini_client import generate_content
from src.utils.prompt_safety import untrusted_text

logger = logging.getLogger(__name__)

MINDMAP_SYSTEM_PROMPT = """Bạn là chuyên gia trực quan hóa tri thức học thuật và sơ đồ tư duy (Mindmap).
Nhiệm vụ: Dựa vào nội dung bài học hoặc tài liệu được cung cấp, hãy tóm lược và tổ chức thành một sơ đồ tư duy phân cấp dạng Markdown, chuẩn cú pháp Markmap.

Quy tắc cấu trúc bắt buộc:
1. Gốc (#): Tên chủ đề hoặc bài học chính (duy nhất 1 dòng # ở dòng đầu tiên).
2. Cấp 1 (##): Các nhánh chính / luận điểm lớn (từ 3 đến 6 nhánh).
3. Cấp 2 (###): Các khái niệm, cơ chế, thành phần hoặc nguyên lý cốt lõi.
4. Cấp 3 (#### hoặc dấu gạch đầu dòng - ): Từ khóa quan trọng, ví dụ thực tế, hoặc lưu ý (mỗi dòng dưới 15 từ).

Yêu cầu xuất ra:
- Trả về DUY NHẤT mã Markdown phân cấp sơ đồ cây.
- TUYỆT ĐỐI KHÔNG bọc trong khối code block (không dùng ```markdown hay ```).
- KHÔNG thêm lời chào, dẫn nhập hay kết luận ngoài nội dung sơ đồ cây.
"""


def generate_mindmap(content: str, title: str = "") -> str:
    """Generate hierarchical Markdown mindmap tree from document/lesson content."""
    safe_title = title.strip() or "Sơ đồ tri thức bài học"
    safe_content = untrusted_text("SOURCE_CONTENT", content, 20_000)

    prompt = f"""Hãy đọc nội dung bài học sau và tạo ra sơ đồ tư duy phân cấp Markdown cho chủ đề: "{safe_title}".

NỘI DUNG THAM CHIẾU:
{safe_content}

Hãy tạo sơ đồ tư duy phân cấp logic, rõ ràng, giúp người học ghi nhớ nhanh nhất:
"""

    try:
        raw_output = generate_content(
            prompt=prompt,
            system_instruction=MINDMAP_SYSTEM_PROMPT,
        ).strip()

        # Clean up any surrounding code block backticks if LLM enclosed them
        cleaned = raw_output
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned).strip()

        # Ensure it has at least a root header
        if not cleaned.startswith("#"):
            cleaned = f"# {safe_title}\n\n" + cleaned

        return cleaned
    except Exception as exc:
        logger.error("Failed to generate mindmap markdown via LLM: %s", exc)
        # Resilient fallback tree structure
        return f"""# {safe_title}
## Tổng quan
### Khái niệm cốt lõi
- Định nghĩa và vai trò trong bài học
### Mục tiêu đạt được
- Nắm vững lý thuyết cơ bản
- Vận dụng vào thực tế
## Kiến thức trọng tâm
### Nguyên lý hoạt động
- Các bước thực hiện tuần tự
- Các lưu ý quan trọng
## Ứng dụng & Tổng kết
### Thực hành
- Xem lại tài liệu và bài tập
"""
