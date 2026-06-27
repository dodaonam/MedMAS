from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from multi_agent.rag_agent.prompt import (
    DECOMPOSE_SYSTEM,
    GENERATE_SYSTEM,
    REWRITE_SYSTEM,
    make_generate_prompt,
)
from multi_agent.rag_agent.state import RAGState
from multi_agent.image_agent.tool import LABEL_VI
from multi_agent.rag_agent.tool import (
    LLM_MODEL,
    expand_context,
    reciprocal_rank_fusion,
    rerank,
    retrieve,
)
from multi_agent.shared_context import QAPair, SourceResult


class DecomposeOutput(BaseModel):
    questions: list[str]


class RewriteOutput(BaseModel):
    clear: str
    synonym: str
    hyde: str


def decompose_node(state: RAGState) -> dict:
    if not state.get("user_question") or state.get("intent") == "media_question":
        return {"sub_questions": []}

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, reasoning_effort="low").with_structured_output(DecomposeOutput)
    result: DecomposeOutput = llm.invoke([
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": state["user_question"]},
    ])
    return {"sub_questions": result.questions}


def rewrite_node(state: RAGState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, reasoning_effort="low").with_structured_output(RewriteOutput)
    queries: list[str] = []

    sub_questions = state["sub_questions"]
    if sub_questions:
        messages_batch = [
            [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": f"Câu hỏi: {q}"}]
            for q in sub_questions
        ]
        results: list[RewriteOutput] = llm.batch(messages_batch)
        for q, result in zip(sub_questions, results):
            queries.append(q)
            queries.append(result.clear)
            queries.append(result.synonym)
            queries.append(result.hyde)

    # Collect unique disease labels from predictions + extracted_findings, then build one query each
    seen_labels: set[str] = set()
    for label, pred in (state.get("predictions") or {}).items():
        if pred["positive"] and label != "No Finding":
            seen_labels.add(label)
    for label, finding in (state.get("extracted_findings") or {}).items():
        if finding["status"] in ("present", "uncertain"):
            seen_labels.add(label)
    for label in seen_labels:
        queries.append(f"Điều trị và tiên lượng bệnh {LABEL_VI.get(label, label)}")

    return {"search_queries": [q for q in queries if q and q.strip()]}


def retrieve_node(state: RAGState) -> dict:
    queries = state["search_queries"]
    if not queries:
        return {"ranked_lists": []}
    ranked_lists = retrieve(queries)
    return {"ranked_lists": ranked_lists}


def rrf_node(state: RAGState) -> dict:
    rrf_docs = reciprocal_rank_fusion(state["ranked_lists"])
    return {"rrf_docs": rrf_docs}


def expand_node(state: RAGState) -> dict:
    expanded = expand_context(state["rrf_docs"])
    return {"expanded_docs": expanded}


def rerank_node(state: RAGState) -> dict:
    query = state.get("user_question") or "X-quang ngực"
    final = rerank(state["expanded_docs"], query)
    return {"final_docs": final}


def generate_node(state: RAGState) -> dict:
    docs = state["final_docs"]
    if state.get("intent") == "media_question" or not state.get("user_question"):
        labels: list[str] = []
        for label, pred in (state.get("predictions") or {}).items():
            if pred["positive"] and label != "No Finding":
                labels.append(LABEL_VI.get(label, label))
        for label, finding in (state.get("extracted_findings") or {}).items():
            if finding["status"] in ("present", "uncertain"):
                vi = LABEL_VI.get(label, label)
                if vi not in labels:
                    labels.append(vi)
        if labels:
            question = f"Điều trị và tiên lượng của: {', '.join(labels)}"
        else:
            question = "Điều trị và tiên lượng các bệnh lý phổi phát hiện trên X-quang ngực."
    else:
        question = state["user_question"]
    prompt = make_generate_prompt(question, docs, "")
    llm = ChatOpenAI(model=LLM_MODEL, streaming=True, temperature=0.3, reasoning_effort="low")
    response = llm.invoke([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": prompt},
    ])

    sources = [
        SourceResult(
            title=d.metadata.get("title", ""),
            url=d.metadata.get("url", ""),
            excerpt=d.page_content[:200],
        )
        for d in docs
    ]
    qa_pair = QAPair(
        question=state.get("user_question") or "",
        answer=response.content,
        sources=sources,
    )
    return {"qa_pairs": [qa_pair]}


def build_rag_agent():
    g = StateGraph(RAGState)
    g.add_node("decompose", decompose_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rrf_fusion", rrf_node)
    g.add_node("expand_context", expand_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rrf_fusion")
    g.add_edge("rrf_fusion", "expand_context")
    g.add_edge("expand_context", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)

    return g.compile()
