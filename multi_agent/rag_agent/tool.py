from __future__ import annotations

import os
import warnings
from collections import defaultdict

import torch
from FlagEmbedding import BGEM3FlagModel
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchAny, MatchValue, Prefetch, SparseVector

# ── Constants ────────────────────────────────────────────────────────────────

RETRIEVE_N = 20
TOP_RRF = 50
TOP_K = 10
RRF_K = 60
LLM_MODEL = "gpt-5.4-mini"
CF_EMBED_MODEL = "@cf/baai/bge-m3"
COHERE_MODEL = "rerank-multilingual-v3.0"
COLLECTION = "medical_docs"

# ── Sparse encoder ────────────────────────────────────────────────────────────

_sparse_model: BGEM3FlagModel | None = None


def get_sparse_model() -> BGEM3FlagModel:
    global _sparse_model
    if _sparse_model is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            use_fp16 = torch.cuda.is_available()
        _sparse_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=use_fp16,
            batch_size=64,
            query_max_length=512,
        )
    return _sparse_model


def encode_queries_sparse(texts: list[str]) -> list[SparseVector]:
    output = get_sparse_model().encode(
        texts,
        batch_size=64,
        max_length=512,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return [_lexical_to_sparse(lw) for lw in output["lexical_weights"]]


def _lexical_to_sparse(lw: dict) -> SparseVector:
    items = sorted([(int(k), float(v)) for k, v in lw.items() if float(v) > 0])
    return SparseVector(
        indices=[i for i, _ in items],
        values=[v for _, v in items],
    )


# ── Qdrant client ─────────────────────────────────────────────────────────────

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*insecure.*")
            _qdrant_client = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=30,
                check_compatibility=False,
            )
    return _qdrant_client


# ── Dense embedder ────────────────────────────────────────────────────────────

_dense_embedder: CloudflareWorkersAIEmbeddings | None = None


def get_dense_embedder() -> CloudflareWorkersAIEmbeddings:
    global _dense_embedder
    if _dense_embedder is None:
        # Support both standard names (CF_ACCOUNT_ID / CF_AI_API_TOKEN) and
        # legacy names used in the current .env (ACCOUNT_ID / CLOUDFLARE_AUTH_TOKEN)
        account_id = os.environ.get("CF_ACCOUNT_ID") or os.environ.get("ACCOUNT_ID", "")
        api_token = os.environ.get("CF_AI_API_TOKEN") or os.environ.get("CLOUDFLARE_AUTH_TOKEN", "")
        _dense_embedder = CloudflareWorkersAIEmbeddings(
            account_id=account_id,
            api_token=api_token,
            model_name=CF_EMBED_MODEL,
        )
    return _dense_embedder


# ── Retrieve ─────────────────────────────────────────────────────────────────

def retrieve(search_queries: list[str]) -> list[list[Document]]:
    """Dense + sparse hybrid search. N Qdrant hybrid calls run in parallel via batch()."""
    dense_vecs = get_dense_embedder().embed_documents(search_queries)
    sparse_vecs = encode_queries_sparse(search_queries)
    client = get_qdrant_client()

    def _hybrid_query(i: int) -> list[Document]:
        points = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vecs[i], using="dense", limit=RETRIEVE_N),
                Prefetch(query=sparse_vecs[i], using="sparse", limit=RETRIEVE_N),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=RETRIEVE_N,
            with_payload=True,
        ).points
        return [_point_to_doc(p) for p in points]

    indices = list(range(len(search_queries)))
    return RunnableLambda(_hybrid_query).batch(
        indices,
        config={"max_concurrency": min(len(indices), 10)},
    )


def _point_to_doc(point) -> Document:
    payload = dict(point.payload or {})
    content = payload.pop("text", "")
    return Document(page_content=content, metadata=payload)


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RRF_K,
    top_n: int = TOP_RRF,
) -> list[Document]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            # Use chunk_id if available, otherwise fall back to url+chunk_index
            doc_id = doc.metadata.get("chunk_id") or (
                str(doc.metadata.get("url", "")) + ":" + str(doc.metadata.get("chunk_index", rank))
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(doc_id, doc)
    sorted_ids = sorted(scores, key=lambda x: -scores[x])
    return [docs[i] for i in sorted_ids[:top_n]]


# ── Context Expand ────────────────────────────────────────────────────────────

def expand_context(rrf_docs: list[Document]) -> list[Document]:
    """Fetch ±1 chunk neighbors from Qdrant, grouped by URL (1 scroll call per URL)."""
    client = get_qdrant_client()
    existing = {
        (d.metadata.get("url"), d.metadata.get("chunk_index"))
        for d in rrf_docs
    }
    url_to_needed: dict[str, set[int]] = defaultdict(set)

    for doc in rrf_docs:
        url = doc.metadata.get("url")
        ci = doc.metadata.get("chunk_index")
        if url is None or ci is None:
            continue
        for neighbor_ci in [ci - 1, ci + 1]:
            if neighbor_ci >= 0 and (url, neighbor_ci) not in existing:
                url_to_needed[url].add(neighbor_ci)

    if not url_to_needed:
        return rrf_docs

    def _scroll(item: tuple[str, set[int]]) -> list[Document]:
        url, needed_indices = item
        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=len(needed_indices),
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(must=[
                FieldCondition(key="url", match=MatchValue(value=url)),
                FieldCondition(key="chunk_index", match=MatchAny(any=list(needed_indices))),
            ]),
        )
        return [_point_to_doc(p) for p in points]

    items = list(url_to_needed.items())
    results = RunnableLambda(_scroll).batch(
        items,
        config={"max_concurrency": min(len(items), 5)},
    )
    extra_docs = [doc for batch in results for doc in batch]
    return rrf_docs + extra_docs


# ── Rerank ────────────────────────────────────────────────────────────────────

def rerank(docs: list[Document], query: str) -> list[Document]:
    reranker = CohereRerank(model=COHERE_MODEL, top_n=TOP_K)
    return reranker.compress_documents(docs, query)
