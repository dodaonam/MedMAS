from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.common import (
    CONTENT_BLOCK_TYPES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHARS,
    DEFAULT_SPLIT_SEPARATORS,
    _normalize_categories,
    _normalize_heading_path,
    _require_text,
    _resolve_article_key,
)


def article_row_to_documents(
    article_row: Mapping[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Sequence[str] = DEFAULT_SPLIT_SEPARATORS,
) -> list[Document]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    title = _require_text(article_row, "title")
    url = _require_text(article_row, "url")
    categories = _normalize_categories(article_row.get("categories", []))
    article_key = _resolve_article_key(article_row, title=title, url=url)

    content_blocks = _prepare_content_blocks(article_row.get("blocks", []), title=title)
    if not content_blocks:
        return []

    runs = _build_runs(content_blocks)
    payloads: list[dict[str, Any]] = []
    for run in runs:
        payloads.extend(
            _run_to_chunk_payloads(
                title=title,
                run=run,
                max_chars=max_chars,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        )

    documents: list[Document] = []
    for chunk_index, payload in enumerate(payloads):
        chunk_id = f"{article_key}::{payload['start_order']}::{payload['end_order']}::{chunk_index}"
        metadata = {
            "url": url,
            "categories": categories[:],
            "title": title,
            "heading_path": payload["heading_path"][:],
            "block_type": payload["block_type"],
            "start_order": payload["start_order"],
            "end_order": payload["end_order"],
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
        }
        documents.append(Document(id=chunk_id, page_content=payload["text"], metadata=metadata))
    return documents


def _prepare_content_blocks(blocks: Any, *, title: str) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        raise ValueError("article_row['blocks'] must be a list")

    prepared: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise ValueError(f"block at position {index} must be a mapping")

        block_type = block.get("type")
        if block_type == "heading":
            continue
        if block_type not in CONTENT_BLOCK_TYPES:
            raise ValueError(f"unsupported block type: {block_type!r}")

        order = block.get("order")
        if not isinstance(order, int):
            raise ValueError(f"block order must be int, got {order!r}")

        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError("block text must be a string")

        original_heading_path = _normalize_heading_path(block.get("heading_path", []))
        prepared.append(
            {
                "block_type": block_type,
                "order": order,
                "text": text.strip(),
                "original_heading_path": original_heading_path,
                "normalized_heading_path": original_heading_path if original_heading_path else [title],
                "has_heading": bool(original_heading_path),
            }
        )

    prepared.sort(key=lambda block: block["order"])
    return prepared


def _build_runs(content_blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for block in content_blocks:
        if not current:
            current = [block]
            continue

        previous = current[-1]
        same_type = block["block_type"] == previous["block_type"]
        same_heading_path = block["normalized_heading_path"] == previous["normalized_heading_path"]
        continuous_order = block["order"] == previous["order"] + 1
        if same_type and same_heading_path and continuous_order:
            current.append(block)
            continue

        runs.append(current)
        current = [block]

    if current:
        runs.append(current)
    return runs


def _run_to_chunk_payloads(
    *,
    title: str,
    run: list[dict[str, Any]],
    max_chars: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> list[dict[str, Any]]:
    if not run:
        return []

    payloads: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []

    for block in run:
        if not current_blocks:
            if len(_render_chunk_text(title=title, blocks=[block])) <= max_chars:
                current_blocks = [block]
            else:
                payloads.extend(
                    _split_single_block(
                        title=title,
                        block=block,
                        max_chars=max_chars,
                        chunk_overlap=chunk_overlap,
                        separators=separators,
                    )
                )
            continue

        candidate_blocks = current_blocks + [block]
        if len(_render_chunk_text(title=title, blocks=candidate_blocks)) <= max_chars:
            current_blocks = candidate_blocks
            continue

        payloads.append(_build_chunk_payload(title=title, blocks=current_blocks))
        if len(_render_chunk_text(title=title, blocks=[block])) <= max_chars:
            current_blocks = [block]
        else:
            payloads.extend(
                _split_single_block(
                    title=title,
                    block=block,
                    max_chars=max_chars,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                )
            )
            current_blocks = []

    if current_blocks:
        payloads.append(_build_chunk_payload(title=title, blocks=current_blocks))
    return payloads


def _split_single_block(
    *,
    title: str,
    block: Mapping[str, Any],
    max_chars: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> list[dict[str, Any]]:
    prefix = _render_prefix(
        title=title,
        heading_path=block["normalized_heading_path"],
        has_heading=bool(block["has_heading"]),
    )
    usable_body_budget = max_chars - len(prefix) - 2
    if usable_body_budget <= 0:
        raise ValueError(
            "prefix alone exceeds or exhausts the configured max_chars budget"
        )

    bounded_overlap = min(chunk_overlap, max(0, usable_body_budget - 1))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=usable_body_budget,
        chunk_overlap=bounded_overlap,
        separators=list(separators),
    )
    split_texts = splitter.split_text(str(block["text"]))
    if not split_texts:
        split_texts = [str(block["text"])]

    payloads: list[dict[str, Any]] = []
    for piece in split_texts:
        payloads.append(
            {
                "text": _render_chunk_text(title=title, blocks=[{**block, "text": piece}]),
                "heading_path": list(block["normalized_heading_path"]),
                "block_type": block["block_type"],
                "start_order": block["order"],
                "end_order": block["order"],
            }
        )
    return payloads


def _build_chunk_payload(*, title: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("cannot build chunk payload from empty block list")

    return {
        "text": _render_chunk_text(title=title, blocks=blocks),
        "heading_path": list(blocks[0]["normalized_heading_path"]),
        "block_type": blocks[0]["block_type"],
        "start_order": blocks[0]["order"],
        "end_order": blocks[-1]["order"],
    }


def _render_chunk_text(*, title: str, blocks: list[dict[str, Any]]) -> str:
    if not blocks:
        raise ValueError("cannot render chunk text from empty block list")

    prefix = _render_prefix(
        title=title,
        heading_path=blocks[0]["normalized_heading_path"],
        has_heading=bool(blocks[0]["has_heading"]),
    )
    body = "\n".join(str(block["text"]).strip() for block in blocks).strip()
    return f"{prefix}\n\n{body}" if body else prefix


def _render_prefix(*, title: str, heading_path: Sequence[str], has_heading: bool) -> str:
    if has_heading:
        return f"{title}\n{' > '.join(heading_path)}"
    return title
