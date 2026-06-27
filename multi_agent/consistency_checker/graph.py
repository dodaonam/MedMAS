from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from multi_agent.consistency_checker.prompt import EXPLAIN_SYSTEM, SYSTEM_PROMPT, make_consistency_prompt, make_explain_prompt
from multi_agent.shared_context import ConsistencyDetail, ConsistencyResult, SharedContext


class ConsistencyDetailItem(BaseModel):
    label: str
    verdict: Literal["no_conflict", "minor_mismatch", "needs_review", "strong_conflict"]
    reason: str


class ConsistencyOutput(BaseModel):
    overall: Literal["no_conflict", "minor_mismatch", "needs_review", "strong_conflict"]
    details: list[ConsistencyDetailItem]


def consistency_node(state: SharedContext) -> dict:
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, reasoning_effort="low").with_structured_output(ConsistencyOutput)
    result: ConsistencyOutput = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_consistency_prompt(
            state["predictions"], state["extracted_findings"]
        )},
    ])
    consistency = ConsistencyResult(
        overall=result.overall,
        details=[
            ConsistencyDetail(label=d.label, verdict=d.verdict, reason=d.reason)
            for d in result.details
        ],
    )
    llm_text = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, reasoning_effort="low")
    explanation = llm_text.invoke([
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": make_explain_prompt(consistency)},
    ])
    return {"consistency": consistency, "consistency_text": explanation.content}


def build_consistency_checker():
    g = StateGraph(SharedContext)
    g.add_node("consistency", consistency_node)
    g.add_edge(START, "consistency")
    g.add_edge("consistency", END)
    return g.compile()
