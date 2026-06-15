from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.common import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHARS,
    DEFAULT_SPLIT_SEPARATORS,
)
from ingestion.indexer import (  # noqa: E402
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    index_documents_to_qdrant_cloud,
)
from ingestion.indexer_local import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    QdrantBGEM3LocalIndexer,
)
from ingestion.loaders import structured_jsonl_to_documents  # noqa: E402


IndexBackend = Literal["endpoint", "local"]


def ingest_structured_jsonl(
    *,
    input_path: Path,
    index_backend: IndexBackend = "endpoint",
    collection_name: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: tuple[str, ...] = DEFAULT_SPLIT_SEPARATORS,
    dense_model: str = DEFAULT_DENSE_MODEL,
    sparse_model: str = DEFAULT_SPARSE_MODEL,
    force_recreate: bool = False,
    batch_size: int = 64,
    wait: bool = True,
) -> dict[str, object]:
    documents = structured_jsonl_to_documents(
        input_path,
        max_chars=max_chars,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    resolved_collection_name = _resolve_collection_name(collection_name)

    if index_backend == "endpoint":
        vector_store = index_documents_to_qdrant_cloud(
            documents,
            collection_name=resolved_collection_name,
            dense_model=dense_model,
            sparse_model=sparse_model,
            force_recreate=force_recreate,
            batch_size=batch_size,
        )
        indexed_count = len(documents)
        resolved_collection_name = vector_store.collection_name
    elif index_backend == "local":
        indexer = QdrantBGEM3LocalIndexer(
            collection_name=resolved_collection_name,
        )
        indexer.ensure_collection(recreate=force_recreate)
        indexed_count = indexer.index_documents(
            documents,
            batch_size=batch_size,
            wait=wait,
        )
        resolved_collection_name = indexer.collection_name
    else:
        raise ValueError(f"Unsupported index backend: {index_backend}")

    article_ids = {
        document.metadata.get("url", document.metadata.get("title", document.id))
        for document in documents
    }
    return {
        "input_path": str(input_path),
        "index_backend": index_backend,
        "collection_name": resolved_collection_name,
        "documents_indexed": indexed_count,
        "articles_indexed": len(article_ids),
        "dense_model": dense_model,
        "sparse_model": sparse_model if index_backend == "endpoint" else "BAAI/bge-m3 lexical_weights",
        "force_recreate": force_recreate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk structured article JSONL and index the resulting documents into "
            "Qdrant using either the endpoint-based hybrid indexer or the local "
            "FlagEmbedding BGE-M3 hybrid indexer."
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to the structured article JSONL file.")
    parser.add_argument(
        "--index-backend",
        choices=("endpoint", "local"),
        default="endpoint",
        help="Select the indexing backend.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection name. Defaults to `QDRANT_COLLECTION` or the local default.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Character overlap used when a single block must be split.",
    )
    parser.add_argument(
        "--dense-model",
        default=DEFAULT_DENSE_MODEL,
        help="Dense embedding model. Used by both backends.",
    )
    parser.add_argument(
        "--sparse-model",
        default=DEFAULT_SPARSE_MODEL,
        help="Sparse embedding model for the endpoint backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when upserting documents into Qdrant.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return before Qdrant confirms writes. Only applies to the local backend.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = ingest_structured_jsonl(
        input_path=args.input_path,
        index_backend=args.index_backend,
        collection_name=args.collection_name,
        max_chars=args.max_chars,
        chunk_overlap=args.chunk_overlap,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        force_recreate=args.force_recreate,
        batch_size=args.batch_size,
        wait=not args.no_wait,
    )

    print("Structured article ingestion complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def _resolve_collection_name(collection_name: str | None) -> str:
    if collection_name:
        return collection_name
    env_collection_name = os.environ.get("QDRANT_COLLECTION")
    if env_collection_name:
        return env_collection_name
    return DEFAULT_COLLECTION_NAME


if __name__ == "__main__":
    raise SystemExit(main())
