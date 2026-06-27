from __future__ import annotations

import asyncio
import contextvars
import io
import json
import threading
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import pypdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fpdf import FPDF

from multi_agent.image_agent.tool import LABEL_VI
from multi_agent.orchestrator.graph import build_orchestrator

load_dotenv()

_DEJAVU_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

app = FastAPI()
_graph = build_orchestrator()
_reports: dict[str, str] = {}


def _extract_pdf_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


_DISCLAIMER = (
    "Tài liệu này chỉ mang tính tham khảo học tập / nghiên cứu.\n"
    "Không phải kết luận y khoa chính thức.\n"
    "Không sử dụng để chẩn đoán hoặc điều trị bệnh."
)


def _build_report_pdf(report_draft: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("DejaVu", fname=_DEJAVU_FONT)
    pdf.add_font("DejaVu", style="B", fname=_DEJAVU_BOLD)

    # ── Title ────────────────────────────────────────────────────────────────
    pdf.set_font("DejaVu", style="B", size=14)
    pdf.cell(0, 10, "MedMAS — Báo cáo X-quang ngực", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    def _separator():
        pdf.set_draw_color(180, 180, 180)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

    def _section(header: str, body: str):
        if not body.strip():
            return
        _separator()
        pdf.set_font("DejaVu", style="B", size=11)
        pdf.cell(0, 8, header, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", size=10)
        for line in body.splitlines():
            pdf.multi_cell(0, 6, line if line.strip() else " ", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # ── Try structured JSON format ────────────────────────────────────────────
    try:
        data = json.loads(report_draft)
        _section("⚠️  TUYÊN BỐ", _DISCLAIMER)
        _section("KỸ THUẬT", data.get("technique", ""))
        _section("MÔ TẢ", data.get("description", ""))
        _section("KẾT LUẬN", data.get("conclusion", ""))
        _section("KHUYẾN NGHỊ", data.get("recommendation", ""))
    except (json.JSONDecodeError, AttributeError):
        # Fallback for legacy plain-text drafts
        _separator()
        pdf.set_font("DejaVu", size=10)
        for line in report_draft.splitlines():
            pdf.multi_cell(0, 6, line or " ", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _fake_stream(text: str):
    """Yield a template string in small chunks to simulate LLM token streaming."""
    chunk_size = 4
    delay = 0.012  # seconds between chunks (~330 chars/s)
    for i in range(0, len(text), chunk_size):
        yield _sse("text", {"content": text[i:i + chunk_size]})
        await asyncio.sleep(delay)


@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.post("/api/new-session")
async def new_session():
    return {"thread_id": str(uuid.uuid4())}


@app.get("/api/report/{thread_id}")
async def get_report(thread_id: str):
    draft = _reports.get(thread_id)
    if not draft:
        return Response(status_code=404)
    return Response(
        content=_build_report_pdf(draft),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="medmas_report_{thread_id[:8]}.pdf"'},
    )


@app.post("/api/chat")
async def chat(
    thread_id: str = Form(...),
    message: str = Form(""),
    image: UploadFile | None = File(None),
    report_pdf: UploadFile | None = File(None),
):
    image_path = None
    report_text = None

    if image and image.filename:
        save_dir = Path(".tmp_uploads")
        save_dir.mkdir(exist_ok=True)
        img_path = save_dir / f"{thread_id}_{image.filename}"
        img_path.write_bytes(await image.read())
        image_path = str(img_path)

    if report_pdf and report_pdf.filename:
        report_text = _extract_pdf_text(await report_pdf.read()) or None

    has_img = image_path is not None
    has_rep = report_text is not None
    has_q   = bool(message)
    if has_img and has_rep:
        scenario = "C-img+report"
    elif has_img:
        scenario = "A-img" + ("+q" if has_q else "-only")
    elif has_rep:
        scenario = "B-report" + ("+q" if has_q else "-only")
    else:
        scenario = "QA"

    invoke_inputs = {
        # Use "" (falsy but not None) when no image uploaded — guarantees the
        # imaging router check fails regardless of LangGraph's None-merge behaviour.
        "image_path": image_path if image_path is not None else "",
        "report_text": report_text,
        "user_question": message or None,
        "response": None,  # clear stale direct-response from checkpoint
    }

    # Reset stale analysis results when new media is uploaded so the router
    # re-runs the relevant agents instead of reusing cached outputs.
    if image_path is not None:
        invoke_inputs["predictions"] = None
        invoke_inputs["gradcam"] = None
    if report_text is not None:
        invoke_inputs["extracted_findings"] = None
        if image_path is None:
            # Report uploaded without a new image: clear previous imaging results
            # so media_response doesn't re-display them and imaging doesn't re-run.
            invoke_inputs["predictions"] = None
            invoke_inputs["gradcam"] = None
    if image_path is not None or report_text is not None:
        invoke_inputs["consistency"] = None
        invoke_inputs["consistency_text"] = None
        invoke_inputs["report_draft"] = None
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"[{scenario}] {(message or '(no text)')[:60]}",
        "metadata": {
            "thread_id": thread_id,
            "scenario": scenario,
            "has_image": has_img,
            "has_report": has_rep,
            "has_question": has_q,
        },
    }

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            for chunk in _graph.stream(
                invoke_inputs,
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                version="v2",
            ):
                asyncio.run_coroutine_threadsafe(q.put(("chunk", chunk)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(("done", None)), loop)

    ctx = contextvars.copy_context()
    threading.Thread(target=lambda: ctx.run(_producer), daemon=True).start()

    async def _stream():
        _gradcam_emitted = False

        while True:
            kind, data = await q.get()

            if kind == "chunk":
                chunk = data

                # ── Token streaming from RAG generate node ────────────────
                if chunk["type"] == "messages":
                    msg, metadata = chunk["data"]
                    if metadata.get("langgraph_node") == "generate":
                        content = msg.content if isinstance(msg.content, str) else ""
                        if content:
                            yield _sse("text", {"content": content})

                # ── Node completion events (root level only) ──────────────
                elif chunk["type"] == "updates" and not chunk["ns"]:
                    for node_name, state_updates in chunk["data"].items():

                        # Direct text responses (greeting, out_of_scope, case_question, media_question)
                        if node_name in ("direct_response", "case_response", "media_response"):
                            resp = state_updates.get("response")
                            if resp:
                                async for chunk in _fake_stream(resp):
                                    yield chunk
                            # Signal frontend to open second bubble immediately after prediction summary
                            if node_name == "media_response":
                                # Re-emit gradcam from state if run_imaging was skipped this turn
                                if not _gradcam_emitted:
                                    for label, b64 in (state_updates.get("gradcam") or {}).items():
                                        yield _sse("gradcam", {"label": LABEL_VI.get(label, label), "b64": b64})
                                yield _sse("media_done", {})

                        # Imaging results — emit immediately when DenseNet finishes
                        elif node_name == "run_imaging":
                            preds = state_updates.get("predictions")
                            gradcam = state_updates.get("gradcam") or {}
                            if preds:
                                yield _sse("imaging_result", {"predictions": {
                                    label: {"score": r["score"], "positive": r["positive"]}
                                    for label, r in preds.items()
                                }})
                            for label, b64 in gradcam.items():
                                yield _sse("gradcam", {"label": LABEL_VI.get(label, label), "b64": b64})
                            if gradcam:
                                _gradcam_emitted = True

                        # NLP results — emit immediately when clinical agent finishes
                        elif node_name == "run_nlp":
                            findings = state_updates.get("extracted_findings")
                            if findings:
                                yield _sse("nlp_result", {"findings": findings})

                        # Consistency results — emit immediately
                        elif node_name == "run_consistency":
                            consistency = state_updates.get("consistency")
                            if consistency:
                                yield _sse("consistency_result", consistency)

                        # RAG sources — emit after RAG completes
                        elif node_name == "run_rag":
                            new_pairs = state_updates.get("qa_pairs") or []
                            if new_pairs and new_pairs[-1].get("sources"):
                                yield _sse("sources", {"sources": new_pairs[-1]["sources"]})

                        # Report ready
                        elif node_name == "run_report":
                            draft = state_updates.get("report_draft")
                            if draft:
                                _reports[thread_id] = draft
                                yield _sse("report", {"thread_id": thread_id})

            elif kind == "error":
                yield _sse("error", {"message": data})
                break

            elif kind == "done":
                yield _sse("done", {})
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")
