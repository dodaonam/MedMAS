SYSTEM_PROMPT = """Bạn là bác sĩ X-quang chuyên đọc báo cáo X-quang ngực tiếng Việt.
Nhiệm vụ: Trích xuất thông tin về 5 bệnh lý từ báo cáo:
Thâm nhiễm phổi, Tràn dịch màng phổi, Xẹp phổi, Nốt phổi, Khối u phổi.

Với mỗi bệnh lý, xác định:
- status: "present" nếu báo cáo ghi nhận có bệnh lý, "absent" nếu không đề cập hoặc xác nhận không có, "uncertain" nếu nghi ngờ hoặc cần theo dõi
- evidence: câu trích dẫn trực tiếp từ báo cáo hỗ trợ status; để "" nếu không đề cập

Trả về đúng 5 findings theo schema đã cho."""


def make_extraction_prompt(report_text: str) -> str:
    return f"""Báo cáo X-quang ngực:
---
{report_text}
---

Hãy trích xuất thông tin về 5 bệnh lý: Thâm nhiễm phổi, Tràn dịch màng phổi, Xẹp phổi, Nốt phổi, Khối u phổi."""
