from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class PredictionResult(TypedDict):
    score: float
    positive: bool  # score > per-label threshold


class FindingResult(TypedDict):
    status: Literal["present", "absent", "uncertain"]
    evidence: str  # direct quote from report; "" if not mentioned


class ConsistencyDetail(TypedDict):
    label: str
    verdict: Literal["no_conflict", "minor_mismatch", "needs_review", "strong_conflict"]
    reason: str


class ConsistencyResult(TypedDict):
    overall: Literal["no_conflict", "minor_mismatch", "needs_review", "strong_conflict"]
    details: list[ConsistencyDetail]


class SourceResult(TypedDict):
    title: str
    url: str
    excerpt: str


class QAPair(TypedDict):
    question: str
    answer: str
    sources: list[SourceResult]


class SharedContext(TypedDict):
    # ── INPUTS ──────────────────────────────────────────────────────────────
    image_path: str | None
    report_text: str | None
    user_question: str | None  # current turn's question (overwritten each turn)

    # ── ORCHESTRATOR ─────────────────────────────────────────────────────────
    intent: NotRequired[str | None]  # set by classify_intent; read by decompose_node

    # ── IMAGING AGENT ───────────────────────────────────────────────────────
    predictions: dict[str, PredictionResult] | None  # 6 labels
    gradcam: dict[str, str] | None                   # base64 PNG per positive label

    # ── CLINICAL NLP AGENT ──────────────────────────────────────────────────
    extracted_findings: dict[str, FindingResult] | None  # 5 disease labels

    # ── CONSISTENCY CHECKER ─────────────────────────────────────────────────
    consistency: ConsistencyResult | None
    consistency_text: NotRequired[str | None]  # LLM-generated explanation

    # ── RAG QA AGENT ────────────────────────────────────────────────────────
    qa_pairs: Annotated[list[QAPair], operator.add]  # append-only reducer

    # ── REPORT DRAFT AGENT ──────────────────────────────────────────────────
    report_draft: str | None
