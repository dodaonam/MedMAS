from __future__ import annotations

from typing import NotRequired

from langchain_core.documents import Document

from multi_agent.shared_context import SharedContext


class RAGState(SharedContext):
    # Internal pipeline fields — not exposed to parent SharedContext
    sub_questions: NotRequired[list[str]]
    search_queries: NotRequired[list[str]]
    ranked_lists: NotRequired[list[list[Document]]]
    rrf_docs: NotRequired[list[Document]]
    expanded_docs: NotRequired[list[Document]]
    final_docs: NotRequired[list[Document]]
