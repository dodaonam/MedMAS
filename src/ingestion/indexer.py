from __future__ import annotations

import os
import uuid
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client.models import Distance


DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_DISTANCE = Distance.COSINE
DEFAULT_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "sparse"


def build_dense_embeddings(
    *,
    model: str = DEFAULT_DENSE_MODEL,
    provider: str | None = None,
    huggingfacehub_api_token: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> HuggingFaceEndpointEmbeddings:
    token = _resolve_huggingface_token(huggingfacehub_api_token)
    resolved_model_kwargs = {"normalize": True, **dict(model_kwargs or {})}
    return HuggingFaceEndpointEmbeddings(
        model=model,
        provider=provider,
        task="feature-extraction",
        model_kwargs=resolved_model_kwargs,
        huggingfacehub_api_token=token,
    )


def build_sparse_embeddings(
    *,
    model_name: str = DEFAULT_SPARSE_MODEL,
    batch_size: int = 256,
    cache_dir: str | None = None,
    threads: int | None = None,
    parallel: int | None = None,
) -> FastEmbedSparse:
    return FastEmbedSparse(
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
        threads=threads,
        parallel=parallel,
    )


def index_documents_to_qdrant_cloud(
    documents: Sequence[Document],
    *,
    collection_name: str,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    huggingfacehub_api_token: str | None = None,
    dense_model: str = DEFAULT_DENSE_MODEL,
    sparse_model: str = DEFAULT_SPARSE_MODEL,
    provider: str | None = None,
    dense_model_kwargs: Mapping[str, Any] | None = None,
    vector_name: str = DEFAULT_VECTOR_NAME,
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
    distance: Distance = DEFAULT_DISTANCE,
    batch_size: int = 64,
    force_recreate: bool = False,
) -> QdrantVectorStore:
    if not documents:
        raise ValueError("documents must not be empty")

    resolved_url = qdrant_url or os.environ.get("QDRANT_URL")
    if not resolved_url:
        raise ValueError("Qdrant URL is required. Set `QDRANT_URL` or pass `qdrant_url`.")

    dense_embeddings = build_dense_embeddings(
        model=dense_model,
        provider=provider,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs=dense_model_kwargs,
    )
    sparse_embeddings = build_sparse_embeddings(model_name=sparse_model)

    prepared_documents = _prepare_documents_for_qdrant(documents)
    return QdrantVectorStore.from_documents(
        list(prepared_documents),
        embedding=dense_embeddings,
        collection_name=collection_name,
        url=resolved_url,
        api_key=qdrant_api_key or os.environ.get("QDRANT_API_KEY"),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=sparse_embeddings,
        distance=distance,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
        force_recreate=force_recreate,
        batch_size=batch_size,
    )


def _resolve_huggingface_token(explicit_token: str | None) -> str:
    token = (
        explicit_token
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        raise ValueError(
            "Hugging Face token is required. Set `HF_TOKEN` or "
            "`HUGGINGFACEHUB_API_TOKEN`, or pass `huggingfacehub_api_token`."
        )
    return token


def _prepare_documents_for_qdrant(documents: Sequence[Document]) -> list[Document]:
    prepared: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        raw_id = document.id
        if raw_id is not None:
            metadata.setdefault("document_id", str(raw_id))

        prepared.append(
            Document(
                id=_to_qdrant_point_id(raw_id),
                page_content=document.page_content,
                metadata=metadata,
            )
        )
    return prepared


def _to_qdrant_point_id(value: str | int | None) -> str | int | None:
    if value is None or isinstance(value, int):
        return value
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(value)))
