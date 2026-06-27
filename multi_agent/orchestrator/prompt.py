from __future__ import annotations

from multi_agent.image_agent.tool import LABEL_VI

INTENT_SYSTEM = """Phân loại ý định của user message thành một trong 5 nhóm:
- greeting: chào hỏi, giới thiệu, hỏi về hệ thống ("Xin chào", "Bạn là gì?", "Bạn làm được gì?")
- medical_question: câu hỏi y khoa tổng quát muốn tra cứu thông tin, KHÔNG liên quan đến ảnh/báo cáo vừa gửi ("Tràn dịch màng phổi là gì?", "Triệu chứng lao phổi?", "Điều trị xẹp phổi như thế nào?")
- case_question: hỏi về kết quả phân tích đã có trong phiên hiện tại, không kèm media mới ("Giải thích kết quả", "Tóm tắt lại cho tôi", "Kết quả phân tích nói gì?")
- media_question: câu hỏi về nội dung ảnh hoặc báo cáo vừa upload, kể cả câu hỏi hỗn hợp về media ("Ảnh này bệnh gì?", "Trong report có bệnh gì?", "Phổi có bị xẹp không?", "Bệnh gì vậy, có nguy hiểm không?", "Đọc ảnh giúp tôi")
- out_of_scope: không liên quan y tế ("Công thức nấu phở", "Thời tiết hôm nay")"""

GREETING_RESPONSE = """Xin chào! Tôi là MedMAS — hệ thống hỗ trợ phân tích X-quang ngực.

Tôi có thể:
- Phân tích ảnh X-quang ngực (tải lên ảnh PNG/JPEG)
- Trích xuất thông tin từ báo cáo X-quang tiếng Việt
- Kiểm tra tính nhất quán giữa ảnh và báo cáo
- Trả lời câu hỏi y khoa dựa trên tài liệu tham khảo
- Tạo bản nháp báo cáo RadReport

Lưu ý: MedMAS chỉ hỗ trợ học tập/nghiên cứu, không thay thế tư vấn y khoa chuyên nghiệp."""

OUT_OF_SCOPE_RESPONSE = "Xin lỗi, tôi chỉ có thể hỗ trợ về X-quang ngực và y khoa. Hãy đặt câu hỏi liên quan đến lĩnh vực này."


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "độ tin cậy cao"
    if score >= 0.75:
        return "độ tin cậy khá cao"
    if score >= 0.65:
        return "độ tin cậy trung bình-khá"
    return "độ tin cậy mức trung bình"


def make_media_response(predictions: dict | None, extracted_findings: dict | None) -> str:
    lines: list[str] = []

    if predictions:
        positive = [
            (label, r) for label, r in predictions.items()
            if r["positive"] and label != "No Finding"
        ]
        negative = [
            (label, r) for label, r in predictions.items()
            if not r["positive"] and label != "No Finding"
        ]

        if positive:
            lines.append("Dựa trên kết quả phân tích ảnh X-quang ngực, có các phát hiện dương tính sau:\n")
            for label, r in sorted(positive, key=lambda x: -x[1]["score"]):
                vi = LABEL_VI.get(label, label)
                lines.append(f"- {vi}: {r['score']*100:.1f}% — {_confidence_label(r['score'])}")
        else:
            lines.append("Dựa trên kết quả phân tích ảnh X-quang ngực, không phát hiện bất thường đáng kể trong ảnh X-quang ngực.")

        if negative:
            lines.append("")
            for label, r in negative:
                vi = LABEL_VI.get(label, label)
                lines.append(f"{vi}: {r['score']*100:.1f}% — âm tính.")

    if extracted_findings:
        status_vi = {"present": "hiện diện", "absent": "không hiện diện", "uncertain": "nghi ngờ"}
        lines.append("\nKết quả trích xuất từ báo cáo X-quang:")
        for label, f in extracted_findings.items():
            vi = LABEL_VI.get(label, label)
            st = status_vi.get(f["status"], f["status"])
            lines.append(f"- {vi}: {st}")

    if not lines:
        return "Chưa có kết quả phân tích ảnh hoặc báo cáo."

    return "\n".join(lines)


def make_case_question_response(state) -> str:
    parts = ["Dưới đây là tóm tắt kết quả phân tích hiện tại:\n"]

    predictions = state.get("predictions")
    if predictions:
        positive = [f"{LABEL_VI.get(l, l)} ({r['score']:.0%})" for l, r in predictions.items() if r["positive"]]
        parts.append(f"**Phân tích ảnh AI:** {', '.join(positive) if positive else 'Không phát hiện bệnh lý'}")

    findings = state.get("extracted_findings")
    if findings:
        present = [LABEL_VI.get(l, l) for l, f in findings.items() if f["status"] == "present"]
        uncertain = [LABEL_VI.get(l, l) for l, f in findings.items() if f["status"] == "uncertain"]
        parts.append(f"**Trích xuất báo cáo:** Hiện diện: {', '.join(present) or 'Không có'}; Nghi ngờ: {', '.join(uncertain) or 'Không có'}")

    consistency = state.get("consistency")
    if consistency:
        parts.append(f"**Consistency:** {consistency['overall']}")

    qa_pairs = state.get("qa_pairs") or []
    if qa_pairs:
        parts.append(f"**Q&A:** {len(qa_pairs)} câu hỏi đã được trả lời")

    report = state.get("report_draft")
    if report:
        parts.append("**Báo cáo nháp:** Đã có")

    if len(parts) == 1:
        return "Chưa có dữ liệu phân tích. Hãy tải lên ảnh X-quang hoặc báo cáo để bắt đầu."

    return "\n".join(parts)
