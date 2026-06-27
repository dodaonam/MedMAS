from multi_agent.image_agent.tool import LABEL_VI

SYSTEM_PROMPT = """Bạn là bác sĩ X-quang chuyên khoa. Tạo bản nháp báo cáo X-quang ngực tiếng Việt.
Chỉ dùng thông tin được cung cấp. Không suy diễn thêm.
Viết plain text, không dùng markdown: không dùng **, ##, *, -."""


def make_report_prompt(
    predictions: dict | None,
    extracted_findings: dict | None,
    consistency,
    qa_pairs: list,
) -> str:
    parts = []

    if predictions:
        positive = [f"{LABEL_VI.get(l, l)} ({r['score']:.0%})" for l, r in predictions.items() if r["positive"]]
        negative = [LABEL_VI.get(l, l) for l, r in predictions.items() if not r["positive"]]
        parts.append(
            f"Phân tích ảnh AI:\n  Dương tính: {', '.join(positive) or 'Không có'}\n"
            f"  Âm tính: {', '.join(negative)}"
        )

    if extracted_findings:
        lines = ["Kết quả trích xuất từ báo cáo:"]
        for label, f in extracted_findings.items():
            vi = LABEL_VI.get(label, label)
            evidence = f" ({f['evidence']})" if f["evidence"] else ""
            status_vi = {"present": "hiện diện", "absent": "không có", "uncertain": "nghi ngờ"}.get(f["status"], f["status"])
            lines.append(f"  {vi}: {status_vi}{evidence}")
        parts.append("\n".join(lines))

    if consistency:
        flag = " ⚠️" if consistency["overall"] in ("needs_review", "strong_conflict") else ""
        parts.append(f"Kiểm tra tính nhất quán: {consistency['overall']}{flag}")
        for d in consistency.get("details", []):
            if d["verdict"] != "no_conflict":
                vi = LABEL_VI.get(d["label"], d["label"])
                parts.append(f"  {vi}: {d['verdict']} — {d['reason']}")

    if qa_pairs:
        parts.append("Thông tin tham khảo từ RAG:")
        for qa in qa_pairs[-2:]:
            parts.append(f"  Q: {qa['question']}\n  A: {qa['answer'][:200]}...")

    data_summary = "\n\n".join(parts) if parts else "Không có dữ liệu phân tích."

    return f"""Dữ liệu phân tích:

{data_summary}

Điền vào các trường sau (plain text, không markdown):

technique: Mô tả kỹ thuật chụp (loại phim, tư thế, chất lượng ảnh).
description: Mô tả chi tiết hình ảnh X-quang dựa trên predictions và extracted_findings.
conclusion: Tổng hợp kết luận lâm sàng. Thêm ⚠️ vào đầu nếu consistency là needs_review hoặc strong_conflict.
recommendation: Khuyến nghị cụ thể (để trống nếu không có finding dương tính và không có conflict)."""
