from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from multi_agent.image_agent.tool import run_inference
from multi_agent.shared_context import SharedContext


def imaging_node(state: SharedContext) -> dict:
    predictions, gradcam = run_inference(state["image_path"])
    return {"predictions": predictions, "gradcam": gradcam}


def build_image_agent():
    g = StateGraph(SharedContext)
    g.add_node("imaging", imaging_node)
    g.add_edge(START, "imaging")
    g.add_edge("imaging", END)
    return g.compile()
