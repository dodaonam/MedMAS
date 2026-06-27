from __future__ import annotations

from typing import Literal, NotRequired

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from multi_agent.clinical_agent.graph import build_clinical_agent
from multi_agent.consistency_checker.graph import build_consistency_checker
from multi_agent.image_agent.graph import build_image_agent
from multi_agent.orchestrator.prompt import (
    GREETING_RESPONSE,
    INTENT_SYSTEM,
    OUT_OF_SCOPE_RESPONSE,
    make_case_question_response,
    make_media_response,
)
from multi_agent.rag_agent.graph import build_rag_agent
from multi_agent.report_agent.graph import build_report_agent
from multi_agent.shared_context import SharedContext


class OrchestratorState(SharedContext):
    # intent is inherited from SharedContext
    response: NotRequired[str | None]  # direct response text (non-agent flows)


class IntentOutput(BaseModel):
    intent: Literal["greeting", "medical_question", "case_question", "media_question", "out_of_scope"]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_intent_node(state: OrchestratorState) -> dict:
    question = state.get("user_question", "")
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, reasoning_effort="low").with_structured_output(IntentOutput)
    result = llm.invoke([
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": question},
    ])
    return {"intent": result.intent}


def direct_response_node(state: OrchestratorState) -> dict:
    intent = state.get("intent")
    if intent == "greeting":
        return {"response": GREETING_RESPONSE}
    return {"response": OUT_OF_SCOPE_RESPONSE}


def case_response_node(state: OrchestratorState) -> dict:
    return {"response": make_case_question_response(state)}


def media_response_node(state: OrchestratorState) -> dict:
    parts = [make_media_response(state.get("predictions"), state.get("extracted_findings"))]
    if state.get("consistency_text"):
        parts.append(f"\nNhận xét tính nhất quán:\n{state['consistency_text']}")
    return {
        "response": "\n".join(parts),
        "gradcam": state.get("gradcam"),
    }


def after_question_node(state: OrchestratorState) -> dict:
    """Clear user_question to prevent re-triggering on subsequent turns."""
    return {"user_question": None}


# ── Routing functions ──────────────────────────────────────────────────────────

def main_router(state: OrchestratorState) -> str:
    """Single router: first runs all needed media analysis, then routes to intent/default."""
    # Phase 1 — media analysis (sequential, loop back after each)
    if state.get("image_path") and state.get("predictions") is None:
        return "run_imaging"
    if state.get("report_text") and state.get("extracted_findings") is None:
        return "run_nlp"
    if (
        state.get("image_path") is not None
        and state.get("report_text") is not None
        and state.get("predictions") is not None
        and state.get("extracted_findings") is not None
        and state.get("consistency") is None
    ):
        return "run_consistency"

    # Phase 2 — intent or default
    if state.get("user_question"):
        return "classify_intent"
    # No text but media present — show prediction summary directly
    if state.get("image_path") or state.get("report_text"):
        return "media_response"
    return "run_rag"


def route_intent(state: OrchestratorState) -> str:
    intent = state.get("intent", "out_of_scope")
    if intent in ("greeting", "out_of_scope"):
        return "direct_response"
    if intent == "case_question":
        return "case_response"
    if intent == "media_question":
        return "media_response"
    return "run_rag"  # medical_question


def after_question_router(state: OrchestratorState) -> str:
    """After question handling: run report if analysis data exists and no draft yet."""
    has_analysis = (
        state.get("predictions") is not None
        or state.get("extracted_findings") is not None
    )
    if has_analysis and state.get("report_draft") is None:
        return "run_report"
    return END


# ── Build ──────────────────────────────────────────────────────────────────────

def build_orchestrator():
    imaging_agent = build_image_agent()
    clinical_agent = build_clinical_agent()
    consistency_checker = build_consistency_checker()
    rag_agent = build_rag_agent()
    report_agent = build_report_agent()

    g = StateGraph(OrchestratorState)

    # Pass-through router node (routing logic lives in conditional edges)
    g.add_node("router", lambda _: {})

    # Media analysis agents
    g.add_node("run_imaging", imaging_agent)
    g.add_node("run_nlp", clinical_agent)
    g.add_node("run_consistency", consistency_checker)

    # Intent & response nodes
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("direct_response", direct_response_node)
    g.add_node("case_response", case_response_node)
    g.add_node("media_response", media_response_node)

    # RAG + report
    g.add_node("run_rag", rag_agent)
    g.add_node("run_report", report_agent)

    # Transition node: clears question and gates report generation
    g.add_node("after_question", after_question_node)

    # ── Edges ────────────────────────────────────────────────────────────────

    g.add_edge(START, "router")

    g.add_conditional_edges("router", main_router, {
        "run_imaging": "run_imaging",
        "run_nlp": "run_nlp",
        "run_consistency": "run_consistency",
        "classify_intent": "classify_intent",
        "media_response": "media_response",
        "run_rag": "run_rag",
    })

    # Media agents loop back to router for next analysis step
    g.add_edge("run_imaging", "router")
    g.add_edge("run_nlp", "router")
    g.add_edge("run_consistency", "router")

    # Intent routing
    g.add_conditional_edges("classify_intent", route_intent, {
        "direct_response": "direct_response",
        "case_response": "case_response",
        "media_response": "media_response",
        "run_rag": "run_rag",
    })

    # Immediate endings — no RAG/report for greetings / case summaries / out_of_scope
    g.add_edge("direct_response", END)
    g.add_edge("case_response", END)

    # media_response emits prediction summary, then continues to RAG for medical context
    g.add_edge("media_response", "run_rag")

    g.add_edge("run_rag", "after_question")

    g.add_conditional_edges("after_question", after_question_router, {
        "run_report": "run_report",
        END: END,
    })

    g.add_edge("run_report", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)
