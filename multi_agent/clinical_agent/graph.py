from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from multi_agent.clinical_agent.prompt import SYSTEM_PROMPT, make_extraction_prompt
from multi_agent.shared_context import FindingResult, SharedContext


class FindingExtraction(BaseModel):
    label: Literal["Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass"]
    status: Literal["present", "absent", "uncertain"]
    evidence: str


class ClinicalNLPOutput(BaseModel):
    findings: list[FindingExtraction]  # always 5 labels


def nlp_node(state: SharedContext) -> dict:
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, reasoning_effort="low").with_structured_output(ClinicalNLPOutput)
    result: ClinicalNLPOutput = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_extraction_prompt(state["report_text"])},
    ])
    extracted_findings = {
        f.label: FindingResult(status=f.status, evidence=f.evidence)
        for f in result.findings
    }
    return {"extracted_findings": extracted_findings}


def build_clinical_agent():
    g = StateGraph(SharedContext)
    g.add_node("nlp", nlp_node)
    g.add_edge(START, "nlp")
    g.add_edge("nlp", END)
    return g.compile()
