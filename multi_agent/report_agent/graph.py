from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from multi_agent.report_agent.prompt import SYSTEM_PROMPT, make_report_prompt
from multi_agent.shared_context import SharedContext


class ReportOutput(BaseModel):
    technique: str
    description: str
    conclusion: str
    recommendation: str


def report_node(state: SharedContext) -> dict:
    prompt = make_report_prompt(
        state.get("predictions"),
        state.get("extracted_findings"),
        state.get("consistency"),
        state.get("qa_pairs") or [],
    )
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, reasoning_effort="low").with_structured_output(ReportOutput)
    result: ReportOutput = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return {"report_draft": result.model_dump_json()}


def build_report_agent():
    g = StateGraph(SharedContext)
    g.add_node("report", report_node)
    g.add_edge(START, "report")
    g.add_edge("report", END)
    return g.compile()
