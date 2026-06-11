from __future__ import annotations

from typing import Any, Mapping, Sequence


CONTENT_BLOCK_TYPES = frozenset({"paragraph", "list_item", "table"})
DEFAULT_MAX_CHARS = 5000
DEFAULT_CHUNK_OVERLAP = 500
DEFAULT_SPLIT_SEPARATORS = ("\n\n", "\n", " ", "")


def _require_text(article_row: Mapping[str, Any], field: str) -> str:
    value = article_row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"article_row[{field!r}] must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"article_row[{field!r}] must not be empty")
    return stripped


def _normalize_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, Sequence):
        raise ValueError("categories must be a sequence of strings")

    categories: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            categories.append(text)
    return categories


def _resolve_article_key(article_row: Mapping[str, Any], *, title: str, url: str) -> str:
    article_id = article_row.get("article_id")
    if isinstance(article_id, str) and article_id.strip():
        return article_id.strip()
    return url or title


def _normalize_heading_path(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("heading_path must be a list")

    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized
