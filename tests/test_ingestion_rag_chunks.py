from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion import article_row_to_documents, structured_jsonl_to_documents


class IngestionRagChunksTests(unittest.TestCase):
    def test_article_row_to_documents_groups_by_type_and_heading_path(self) -> None:
        article = {
            "article_id": "article-1",
            "url": "https://example.com/a",
            "categories": ["Ortho"],
            "title": "Article Title",
            "blocks": [
                {"type": "paragraph", "text": "intro one", "heading_path": [], "order": 1},
                {"type": "paragraph", "text": "intro two", "heading_path": [], "order": 2},
                {"type": "heading", "text": "Section A", "heading_path": ["Section A"], "order": 3, "level": 2},
                {"type": "paragraph", "text": "body one", "heading_path": ["Section A"], "order": 4},
                {"type": "paragraph", "text": "body two", "heading_path": ["Section A"], "order": 5},
                {"type": "list_item", "text": "item one", "heading_path": ["Section A"], "order": 6},
                {"type": "list_item", "text": "item two", "heading_path": ["Section A"], "order": 7},
                {"type": "paragraph", "text": "body three", "heading_path": ["Section A"], "order": 8},
            ],
        }

        documents = article_row_to_documents(article)

        self.assertEqual(len(documents), 4)
        self.assertEqual(
            documents[0].page_content,
            "Article Title\n\nintro one\nintro two",
        )
        self.assertEqual(documents[0].metadata["heading_path"], ["Article Title"])
        self.assertEqual(documents[0].metadata["block_type"], "paragraph")
        self.assertEqual(documents[0].metadata["start_order"], 1)
        self.assertEqual(documents[0].metadata["end_order"], 2)

        self.assertEqual(
            documents[1].page_content,
            "Article Title\nSection A\n\nbody one\nbody two",
        )
        self.assertEqual(documents[1].metadata["heading_path"], ["Section A"])
        self.assertEqual(documents[1].metadata["block_type"], "paragraph")
        self.assertEqual(documents[1].metadata["start_order"], 4)
        self.assertEqual(documents[1].metadata["end_order"], 5)

        self.assertEqual(
            documents[2].page_content,
            "Article Title\nSection A\n\nitem one\nitem two",
        )
        self.assertEqual(documents[2].metadata["block_type"], "list_item")
        self.assertEqual(documents[2].metadata["start_order"], 6)
        self.assertEqual(documents[2].metadata["end_order"], 7)

        self.assertEqual(
            documents[3].page_content,
            "Article Title\nSection A\n\nbody three",
        )
        self.assertEqual(documents[3].metadata["start_order"], 8)
        self.assertEqual(documents[3].metadata["end_order"], 8)
        self.assertEqual(
            [document.metadata["chunk_index"] for document in documents],
            [0, 1, 2, 3],
        )

    def test_continuous_order_is_required_for_grouping(self) -> None:
        article = {
            "article_id": "article-2",
            "url": "https://example.com/b",
            "categories": [],
            "title": "Gap Example",
            "blocks": [
                {"type": "paragraph", "text": "alpha", "heading_path": ["Section"], "order": 1},
                {"type": "paragraph", "text": "beta", "heading_path": ["Section"], "order": 3},
            ],
        }

        documents = article_row_to_documents(article)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["start_order"], 1)
        self.assertEqual(documents[0].metadata["end_order"], 1)
        self.assertEqual(documents[1].metadata["start_order"], 3)
        self.assertEqual(documents[1].metadata["end_order"], 3)

    def test_greedy_flush_splits_run_left_to_right(self) -> None:
        article = {
            "article_id": "article-3",
            "url": "https://example.com/c",
            "categories": ["Test"],
            "title": "T",
            "blocks": [
                {"type": "paragraph", "text": "a" * 10, "heading_path": ["H"], "order": 1},
                {"type": "paragraph", "text": "b" * 10, "heading_path": ["H"], "order": 2},
                {"type": "paragraph", "text": "c" * 10, "heading_path": ["H"], "order": 3},
            ],
        }

        documents = article_row_to_documents(article, max_chars=30)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["start_order"], 1)
        self.assertEqual(documents[0].metadata["end_order"], 2)
        self.assertEqual(documents[1].metadata["start_order"], 3)
        self.assertEqual(documents[1].metadata["end_order"], 3)
        self.assertLessEqual(len(documents[0].page_content), 30)
        self.assertLessEqual(len(documents[1].page_content), 30)

    def test_single_oversized_block_is_split_and_preserves_order(self) -> None:
        article = {
            "article_id": "article-4",
            "url": "https://example.com/d",
            "categories": ["Test"],
            "title": "Title",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
                    "heading_path": [],
                    "order": 1,
                }
            ],
        }

        documents = article_row_to_documents(article, max_chars=30, chunk_overlap=5)

        self.assertGreater(len(documents), 1)
        for index, document in enumerate(documents):
            self.assertEqual(document.metadata["start_order"], 1)
            self.assertEqual(document.metadata["end_order"], 1)
            self.assertEqual(document.metadata["heading_path"], ["Title"])
            self.assertEqual(document.metadata["block_type"], "paragraph")
            self.assertEqual(document.metadata["chunk_index"], index)
            self.assertEqual(document.id, document.metadata["chunk_id"])
            self.assertTrue(document.page_content.startswith("Title\n\n"))
            self.assertLessEqual(len(document.page_content), 30)

    def test_structured_jsonl_to_documents_reads_articles(self) -> None:
        article = {
            "article_id": "article-5",
            "url": "https://example.com/e",
            "categories": ["Test"],
            "title": "Loaded Title",
            "content_hash": "hash",
            "raw_text": "hello",
            "breadcrumb": ["Home"],
            "blocks": [
                {"type": "paragraph", "text": "hello", "heading_path": [], "order": 1},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "articles.jsonl"
            pd.DataFrame([article]).to_json(path, orient="records", lines=True, force_ascii=False)
            documents = structured_jsonl_to_documents(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["title"], "Loaded Title")
        self.assertEqual(documents[0].page_content, "Loaded Title\n\nhello")


if __name__ == "__main__":
    unittest.main()
