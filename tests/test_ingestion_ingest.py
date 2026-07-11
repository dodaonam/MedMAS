from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.ingest import ingest_structured_jsonl


class IngestionEntrypointTests(unittest.TestCase):
    def test_ingest_structured_jsonl_uses_local_backend(self) -> None:
        documents = [
            Document(
                id="doc-2",
                page_content="Kho tho va dau nguc",
                metadata={"url": "https://example.com/b", "title": "B"},
            )
        ]

        with (
            patch("ingestion.ingest.structured_jsonl_to_documents", return_value=documents),
            patch("ingestion.ingest.QdrantBGEM3LocalIndexer") as local_indexer_cls,
        ):
            indexer = local_indexer_cls.return_value
            indexer.collection_name = "local-collection"
            indexer.index_documents.return_value = 1

            summary = ingest_structured_jsonl(
                input_path=Path("articles.jsonl"),
                collection_name="local-collection",
                force_recreate=True,
                batch_size=16,
                wait=False,
            )

        local_indexer_cls.assert_called_once_with(collection_name="local-collection")
        indexer.ensure_collection.assert_called_once_with(recreate=True)
        indexer.index_documents.assert_called_once_with(
            documents,
            batch_size=16,
            wait=False,
        )
        self.assertEqual(summary["collection_name"], "local-collection")
        self.assertEqual(summary["documents_indexed"], 1)
        self.assertEqual(summary["articles_indexed"], 1)
        self.assertTrue(summary["force_recreate"])


if __name__ == "__main__":
    unittest.main()
