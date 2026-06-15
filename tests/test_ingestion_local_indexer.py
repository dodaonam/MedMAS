from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import models


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.indexer_local import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME,
    LocalBGEM3Embedder,
    QdrantBGEM3LocalIndexer,
)


class LocalIndexerTests(unittest.TestCase):
    def test_lexical_to_sparse_vector_sorts_and_filters(self) -> None:
        sparse = LocalBGEM3Embedder.lexical_to_sparse_vector(
            {
                "10": 0.5,
                "2": 1.25,
                "bad": 7,
                "4": 0,
                "7": -0.5,
            }
        )

        self.assertEqual(sparse.indices, [2, 10])
        self.assertEqual(sparse.values, [1.25, 0.5])

    def test_ensure_collection_creates_expected_schema(self) -> None:
        client = _FakeClient()
        indexer = QdrantBGEM3LocalIndexer(
            collection_name="medical_docs",
            client=client,
            embedder=_FakeEmbedder(),
        )

        indexer.ensure_collection(recreate=False)

        self.assertEqual(client.created_collection_name, "medical_docs")
        dense = client.created_vectors_config[DEFAULT_DENSE_VECTOR_NAME]
        sparse = client.created_sparse_vectors_config[DEFAULT_SPARSE_VECTOR_NAME]
        self.assertEqual(dense.size, DEFAULT_DENSE_DIMENSION)
        self.assertEqual(dense.distance, models.Distance.COSINE)
        self.assertIsInstance(sparse, models.SparseVectorParams)

    def test_index_documents_upserts_dense_and_sparse_vectors(self) -> None:
        client = _FakeClient()
        indexer = QdrantBGEM3LocalIndexer(
            collection_name="medical_docs",
            client=client,
            embedder=_FakeEmbedder(),
        )
        documents = [
            Document(
                id="article-1::1::1::0",
                page_content="Ho sot va kho tho",
                metadata={"title": "Resp", "chunk_index": 0},
            )
        ]

        indexed_count = indexer.index_documents(documents, batch_size=8, wait=True)

        self.assertEqual(indexed_count, 1)
        self.assertEqual(len(client.upserts), 1)
        upsert = client.upserts[0]
        self.assertEqual(upsert["collection_name"], "medical_docs")
        point = upsert["points"][0]
        self.assertEqual(
            point.id,
            str(uuid.uuid5(uuid.NAMESPACE_URL, "article-1::1::1::0")),
        )
        self.assertEqual(point.vector[DEFAULT_DENSE_VECTOR_NAME], [0.1, 0.2, 0.3])
        self.assertEqual(point.vector[DEFAULT_SPARSE_VECTOR_NAME].indices, [1, 5])
        self.assertEqual(point.vector[DEFAULT_SPARSE_VECTOR_NAME].values, [0.25, 0.75])
        self.assertEqual(point.payload["document_id"], "article-1::1::1::0")
        self.assertEqual(point.payload["text"], "Ho sot va kho tho")
        self.assertEqual(point.payload["title"], "Resp")


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[dict[str, object]]:
        return [
            {
                "dense": [0.1, 0.2, 0.3],
                "sparse": models.SparseVector(indices=[1, 5], values=[0.25, 0.75]),
            }
            for _ in texts
        ]


class _FakeClient:
    def __init__(self) -> None:
        self.exists = False
        self.created_collection_name: str | None = None
        self.created_vectors_config = None
        self.created_sparse_vectors_config = None
        self.upserts: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: dict[str, models.VectorParams],
        sparse_vectors_config: dict[str, models.SparseVectorParams],
    ) -> bool:
        self.created_collection_name = collection_name
        self.created_vectors_config = vectors_config
        self.created_sparse_vectors_config = sparse_vectors_config
        self.exists = True
        return True

    def delete_collection(self, collection_name: str) -> bool:
        self.exists = False
        return True

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[models.PointStruct],
        wait: bool,
    ) -> bool:
        self.upserts.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
            }
        )
        return True


if __name__ == "__main__":
    unittest.main()
