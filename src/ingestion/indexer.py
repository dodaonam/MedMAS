from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


load_dotenv()


DEFAULT_COLLECTION_NAME = "medical_docs"
DEFAULT_MODEL_NAME = "nqd-301125/bge-m3-medical-vi-dense"
DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "sparse"
DEFAULT_DENSE_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 64
DEFAULT_QUERY_MAX_LENGTH = 512
DEFAULT_PASSAGE_MAX_LENGTH = 1024
DEFAULT_QDRANT_TIMEOUT = 360


class LocalBGEM3Embedder:
    """Local BGE-M3 embedder that returns dense + sparse representations."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        use_fp16: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_max_length: int = DEFAULT_QUERY_MAX_LENGTH,
        passage_max_length: int = DEFAULT_PASSAGE_MAX_LENGTH,
        devices: str | list[str] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "Local BGE-M3 indexing requires `FlagEmbedding` and `torch` to be "
                "installed in the current environment."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.use_fp16 = bool(use_fp16 and torch.cuda.is_available())

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=self.use_fp16,
            devices=devices,
            cache_dir=cache_dir,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    @staticmethod
    def lexical_to_sparse_vector(lexical_weights: Mapping[Any, Any]) -> models.SparseVector:
        items: list[tuple[int, float]] = []
        for token_id, weight in lexical_weights.items():
            try:
                index = int(token_id)
                value = float(weight)
            except (TypeError, ValueError):
                continue
            if value > 0:
                items.append((index, value))

        items.sort(key=lambda item: item[0])
        return models.SparseVector(
            indices=[index for index, _ in items],
            values=[value for _, value in items],
        )

    def embed_documents(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        clean_texts = [self._require_text(text) for text in texts]
        if not clean_texts:
            return []

        output = self.model.encode_corpus(
            list(clean_texts),
            batch_size=self.batch_size,
            max_length=self.passage_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        embeddings: list[dict[str, Any]] = []
        for dense_vec, lexical_weight in zip(dense_vecs, lexical_weights, strict=True):
            embeddings.append(
                {
                    "dense": dense_vec.tolist(),
                    "sparse": self.lexical_to_sparse_vector(lexical_weight),
                }
            )
        return embeddings

    @staticmethod
    def _require_text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class QdrantBGEM3LocalIndexer:
    """Hybrid indexer for local BGE-M3 embeddings and Qdrant Cloud."""

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        qdrant_timeout: int | None = None,
        dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
        sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
        embedder: LocalBGEM3Embedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name or os.environ.get(
            "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME
        )
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.qdrant_timeout = qdrant_timeout or _env_int(
            "QDRANT_TIMEOUT", DEFAULT_QDRANT_TIMEOUT
        )

        if client is None:
            resolved_url = qdrant_url or os.environ.get("QDRANT_URL")
            if not resolved_url:
                raise ValueError(
                    "Qdrant URL is required. Set `QDRANT_URL` or pass `qdrant_url`."
                )
            client = QdrantClient(
                url=resolved_url,
                api_key=qdrant_api_key or os.environ.get("QDRANT_API_KEY"),
                timeout=self.qdrant_timeout,
            )
        self.client = client

        self.embedder = embedder or LocalBGEM3Embedder(
            model_name=os.environ.get("BGE_MODEL_NAME", DEFAULT_MODEL_NAME),
            use_fp16=_env_bool("BGE_USE_FP16", True),
            batch_size=_env_int("BGE_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            query_max_length=_env_int(
                "BGE_QUERY_MAX_LENGTH", DEFAULT_QUERY_MAX_LENGTH
            ),
            passage_max_length=_env_int(
                "BGE_PASSAGE_MAX_LENGTH", DEFAULT_PASSAGE_MAX_LENGTH
            ),
        )

    def ensure_collection(self, *, recreate: bool = False) -> None:
        vectors_config = {
            self.dense_vector_name: models.VectorParams(
                size=DEFAULT_DENSE_DIMENSION,
                distance=models.Distance.COSINE,
            )
        }
        sparse_vectors_config = {
            self.sparse_vector_name: models.SparseVectorParams()
        }

        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 32,
        wait: bool = True,
    ) -> int:
        chunks = [
            {
                "id": document.id,
                "text": document.page_content,
                "metadata": dict(document.metadata),
            }
            for document in documents
        ]
        return self.index_chunks(chunks, batch_size=batch_size, wait=wait)

    def index_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 32,
        wait: bool = True,
    ) -> int:
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            texts = [self._require_chunk_text(chunk) for chunk in batch]
            embeddings = self.embedder.embed_documents(texts)

            points = [
                models.PointStruct(
                    id=self._to_qdrant_point_id(chunk.get("id")),
                    vector={
                        self.dense_vector_name: embedding["dense"],
                        self.sparse_vector_name: embedding["sparse"],
                    },
                    payload=self._build_payload(chunk, text),
                )
                for chunk, text, embedding in zip(batch, texts, embeddings, strict=True)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=wait,
                timeout=self.qdrant_timeout,
            )
            total += len(points)

        return total

    @staticmethod
    def _require_chunk_text(chunk: Mapping[str, Any]) -> str:
        text = chunk.get("text")
        if not isinstance(text, str):
            raise ValueError("chunk['text'] must be a string")
        stripped = text.strip()
        if not stripped:
            raise ValueError("chunk['text'] must not be empty")
        return stripped

    @staticmethod
    def _to_qdrant_point_id(value: Any) -> str | int:
        if isinstance(value, int):
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, str) and value.strip():
            raw_value = value.strip()
            try:
                return str(uuid.UUID(raw_value))
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_value))
        return str(uuid.uuid4())

    @staticmethod
    def _build_payload(chunk: Mapping[str, Any], text: str) -> dict[str, Any]:
        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("chunk['metadata'] must be a mapping when provided")

        payload = dict(metadata)
        chunk_id = chunk.get("id")
        if chunk_id is not None:
            payload.setdefault("document_id", str(chunk_id))
        payload["text"] = text
        return payload


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)
