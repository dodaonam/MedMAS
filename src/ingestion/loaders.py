from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd
from langchain_core.documents import Document

from ingestion.chunking import article_row_to_documents
from ingestion.common import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHARS,
    DEFAULT_SPLIT_SEPARATORS,
)


def load_structured_articles(path: str | Path) -> pd.DataFrame:
    return pd.read_json(Path(path), lines=True)


def structured_jsonl_to_documents(
    path: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] = DEFAULT_SPLIT_SEPARATORS,
) -> list[Document]:
    return structured_articles_to_documents(
        load_structured_articles(path),
        max_chars=max_chars,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def structured_articles_to_documents(
    articles: pd.DataFrame,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] = DEFAULT_SPLIT_SEPARATORS,
) -> list[Document]:
    documents: list[Document] = []
    for _, row in articles.reset_index(drop=True).iterrows():
        article_row = cast(Mapping[str, Any], row.to_dict())
        documents.extend(
            article_row_to_documents(
                article_row,
                max_chars=max_chars,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        )
    return documents
