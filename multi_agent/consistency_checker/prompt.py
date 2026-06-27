from multi_agent.image_agent.tool import LABEL_VI

EXPLAIN_SYSTEM = """Bạn là bác sĩ X-quang. Dựa trên kết quả kiểm tra tính nhất quán giữa model AI và báo cáo lâm sàng, hãy viết một nhận xét ngắn gọn bằng tiếng Việt thuần túy (không dùng markdown, không dùng tên bệnh tiếng Anh).
Nêu rõ: mức độ nhất quán tổng thể, các điểm mâu thuẫn cụ thể nếu có, và khuyến nghị hành động."""

SYSTEM_PROMPT = """Bạn là bác sĩ X-quang kiểm tra tính nhất quán giữa kết quả model AI và báo cáo lâm sàng.
So sánh từng bệnh lý trong 5 nhãn: Thâm nhiễm phổi, Tràn dịch màng phổi, Xẹp phổi, Nốt phổi, Khối u phổi.

Hướng dẫn verdict:
- "no_conflict": model và báo cáo đồng thuận (cả hai dương tính, hoặc cả hai âm tính)
- "minor_mismatch": một bên uncertain, không mâu thuẫn rõ ràng
- "needs_review": có điểm nghi ngờ đáng xem xét lại
- "strong_conflict": model dương tính nhưng báo cáo ghi absent, hoặc ngược lại

Overall: verdict tệ nhất trong tất cả labels hoặc đánh giá tổng thể bức tranh."""


def make_consistency_prompt(predictions: dict, extracted_findings: dict) -> str:
    disease_labels = ["Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass"]
    lines = ["So sánh kết quả phân tích:\n"]
    lines.append(f"{'Bệnh lý':<30} {'Model AI':<12} {'Báo cáo':<12} Bằng chứng")
    lines.append("-" * 90)
    for label in disease_labels:
        vi = LABEL_VI.get(label, label)
        pred = predictions.get(label, {})
        finding = extracted_findings.get(label, {})
        model_result = "dương tính" if pred.get("positive") else "âm tính"
        nlp_status = {"present": "hiện diện", "absent": "không có", "uncertain": "nghi ngờ"}.get(
            finding.get("status", "absent"), "không có"
        )
        evidence = (finding.get("evidence", "") or "")[:80]
        lines.append(f"{vi:<30} {model_result:<12} {nlp_status:<12} {evidence}")
    return "\n".join(lines)


def make_explain_prompt(consistency: dict) -> str:
    verdict_vi = {
        "no_conflict": "nhất quán hoàn toàn",
        "minor_mismatch": "có sự khác biệt nhỏ",
        "needs_review": "cần xem xét lại",
        "strong_conflict": "mâu thuẫn rõ ràng",
    }
    lines = [f"Kết quả tổng thể: {verdict_vi.get(consistency['overall'], consistency['overall'])}\n"]
    lines.append("Chi tiết từng bệnh lý:")
    for d in consistency.get("details", []):
        vi = LABEL_VI.get(d["label"], d["label"])
        v = verdict_vi.get(d["verdict"], d["verdict"])
        lines.append(f"- {vi}: {v} — {d['reason']}")
    return "\n".join(lines)
