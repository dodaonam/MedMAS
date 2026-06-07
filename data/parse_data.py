from bs4 import BeautifulSoup, Tag
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DATA_DIR = Path(__file__).resolve().parent
RAW_HTML_INPUT_PATH = DATA_DIR / "raw" / "vinmec_html.jsonl"
PROCESSED_DIR = DATA_DIR / "proccessed"
STRUCTURED_OUTPUT_PATH = PROCESSED_DIR / "vinmec_articles_structured.jsonl"
PARSE_REPORT_OUTPUT_PATH = PROCESSED_DIR / "vinmec_parse_report.json"
RAW_LOAD_PROGRESS_EVERY = 10000
PARSE_PROGRESS_EVERY = 500

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "mc_cid",
    "mc_eid",
    "spm",
    "yclid",
}

NOISY_CONTAINER_HINTS = {
    "table-of-contents",
    "toc",
    "social",
    "share",
    "booking",
    "book-appointment",
    "appointment",
    "dat-lich",
    "lich-kham",
    "hotline",
    "banner",
    "advert",
    "ads",
    "related",
    "recommended",
    "consult",
    "widget",
}

UNWANTED_TAGS = {
    "script",
    "style",
    "iframe",
    "noscript",
    "svg",
    "canvas",
    "form",
    "input",
    "button",
    "aside",
    "footer",
    "header",
    "nav",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"để đặt lịch khám tại viện", re.IGNORECASE),
    re.compile(r"tải và đặt lịch khám tự động trên\s*ứng dụng\s*myvinmec", re.IGNORECASE),
    re.compile(r"đặt lịch trực tiếp\s*tại đây", re.IGNORECASE),
    re.compile(r"quý khách vui lòng bấm số\s*hotline", re.IGNORECASE),
    re.compile(r"gọi tổng đài", re.IGNORECASE),
    re.compile(r"có thể bạn quan tâm", re.IGNORECASE),
    re.compile(r"bài viết liên quan", re.IGNORECASE),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw

    filtered_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_"):
            continue
        if lowered in TRACKING_QUERY_KEYS:
            continue
        filtered_pairs.append((key, value))

    normalized_query = urlencode(filtered_pairs, doseq=True)
    normalized_path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            normalized_query,
            "",
        )
    )


def normalize_inline_text(text: str) -> str:
    normalized = (text or "").replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_block_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_for_hash(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def is_boilerplate_text(text: str) -> bool:
    candidate = normalize_inline_text(text)
    if not candidate:
        return True
    if len(candidate) < 3:
        return True

    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(candidate):
            return True
    return False


def extract_meta_content(soup: BeautifulSoup, *, prop: str | None = None, name: str | None = None) -> str | None:
    meta_tag: Tag | None = None
    if prop is not None:
        meta_tag = soup.find("meta", attrs={"property": prop})
    if meta_tag is None and name is not None:
        meta_tag = soup.find("meta", attrs={"name": name})
    if meta_tag is None:
        return None

    content = meta_tag.get("content")
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    return stripped or None


def extract_json_ld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        payload = (script.string or script.text or "").strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            objects.append(parsed)
        elif isinstance(parsed, list):
            objects.extend(item for item in parsed if isinstance(item, dict))
    return objects


def extract_ld_breadcrumb(soup: BeautifulSoup) -> list[str]:
    for root in extract_json_ld_objects(soup):
        graph_nodes = root.get("@graph") if isinstance(root, dict) else None
        if isinstance(graph_nodes, list):
            nodes = [node for node in graph_nodes if isinstance(node, dict)]
        else:
            nodes = [root]

        for node in nodes:
            if node.get("@type") != "BreadcrumbList":
                continue

            raw_elements = node.get("itemListElement")
            if not isinstance(raw_elements, list):
                continue

            breadcrumb: list[str] = []
            for element in raw_elements:
                if not isinstance(element, dict):
                    continue
                item = element.get("item")
                name = item.get("name") if isinstance(item, dict) else None
                if isinstance(name, str) and name.strip():
                    breadcrumb.append(name.strip())
            if breadcrumb:
                return breadcrumb
    return []


def article_url_from_soup(soup: BeautifulSoup, fallback_url: str) -> str:
    link_tag = soup.find("link", attrs={"rel": "canonical"})
    if link_tag is not None:
        href = link_tag.get("href")
        if isinstance(href, str) and href.strip():
            return normalize_url(urljoin(fallback_url, href.strip()))

    og_url = extract_meta_content(soup, prop="og:url")
    if og_url:
        return normalize_url(urljoin(fallback_url, og_url))

    return normalize_url(fallback_url)


def choose_main_content_node(soup: BeautifulSoup) -> Tag | None:
    selectors = [
        "div#main-article.entry",
        "div#main-article",
        "article",
        "main article",
        "main",
        "div.entry",
    ]

    candidates: list[Tag] = []
    seen_ids: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            if isinstance(node, Tag):
                node_id = id(node)
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    candidates.append(node)

    if not candidates:
        return None

    return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))


def update_heading_path(current_path: list[str], level: int, heading_text: str) -> list[str]:
    depth = max(level - 2, 0)
    next_path = current_path[:depth]
    next_path.append(heading_text)
    return next_path


def table_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_inline_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in rows]

    header = padded_rows[0]
    separator = ["---"] * max_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def should_remove_noisy_node(node: Tag) -> bool:
    raw_parts: list[str] = []
    node_id = node.attrs.get("id")
    if isinstance(node_id, str) and node_id.strip():
        raw_parts.append(node_id.strip().lower())

    node_classes = node.attrs.get("class")
    if isinstance(node_classes, list):
        for cls in node_classes:
            if isinstance(cls, str) and cls.strip():
                raw_parts.append(cls.strip().lower())

    if not raw_parts:
        return False

    token_set: set[str] = set()
    for part in raw_parts:
        token_set.update(token for token in re.split(r"[^a-z0-9]+", part) if token)

    for hint in NOISY_CONTAINER_HINTS:
        normalized_hint = hint.lower()
        if normalized_hint in raw_parts:
            return True
        if "-" in normalized_hint:
            hint_tokens = [token for token in normalized_hint.split("-") if token]
            if hint_tokens and all(token in token_set for token in hint_tokens):
                return True
        elif normalized_hint in token_set:
            return True

    return False


def strip_noisy_nodes(content_node: Tag) -> None:
    for noisy in content_node.find_all(UNWANTED_TAGS):
        noisy.decompose()

    for node in list(content_node.find_all(True)):
        if not isinstance(getattr(node, "attrs", None), dict):
            continue

        if should_remove_noisy_node(node):
            node.decompose()


def extract_blocks(content_node: Tag) -> list[dict[str, Any]]:
    strip_noisy_nodes(content_node)

    blocks: list[dict[str, Any]] = []
    heading_path: list[str] = []
    order = 0

    for element in content_node.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        tag_name = element.name
        if tag_name is None:
            continue

        if tag_name != "table" and element.find_parent("table") is not None:
            continue
        if tag_name == "table" and element.find_parent("table") is not None:
            continue

        if tag_name.startswith("h"):
            heading_text = normalize_inline_text(element.get_text(" ", strip=True))
            if is_boilerplate_text(heading_text):
                continue

            heading_level = int(tag_name[1])
            heading_path = update_heading_path(heading_path, heading_level, heading_text)
            order += 1
            blocks.append(
                {
                    "type": "heading",
                    "level": heading_level,
                    "text": heading_text,
                    "heading_path": heading_path.copy(),
                    "order": order,
                }
            )
            continue

        if tag_name == "table":
            table_md = table_to_markdown(element)
            if is_boilerplate_text(table_md):
                continue
            order += 1
            blocks.append(
                {
                    "type": "table",
                    "text": table_md,
                    "heading_path": heading_path.copy(),
                    "order": order,
                }
            )
            continue

        text = normalize_inline_text(element.get_text(" ", strip=True))
        if is_boilerplate_text(text):
            continue

        block_type = "paragraph" if tag_name == "p" else "list_item"
        order += 1
        blocks.append(
            {
                "type": block_type,
                "text": text,
                "heading_path": heading_path.copy(),
                "order": order,
            }
        )

    return blocks


def render_raw_text(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return normalize_block_text("\n".join(lines))


def extract_article_links(index_html: str, index_base_url: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not isinstance(href, str) or "/bai-viet/" not in href:
            continue

        absolute_url = normalize_url(urljoin(index_base_url, href))
        parsed = urlsplit(absolute_url)
        if not parsed.netloc.endswith("vinmec.com"):
            continue
        if "/bai-viet/" not in parsed.path:
            continue

        if absolute_url in seen:
            continue

        seen.add(absolute_url)
        links.append(absolute_url)

    return links


def parse_article(html: str, url: str, category: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")

    title_node = soup.find("h1", class_="single-title single-title-line") or soup.find("h1")
    title = normalize_inline_text(title_node.get_text(" ", strip=True)) if title_node else ""
    if not title:
        title = normalize_inline_text(extract_meta_content(soup, prop="og:title") or "")
    if not title:
        return None

    content_node = choose_main_content_node(soup)
    if content_node is None:
        return None

    blocks = extract_blocks(content_node)
    if not blocks:
        return None

    raw_text = render_raw_text(blocks)
    if not raw_text:
        return None

    article_url = article_url_from_soup(soup, url)
    breadcrumb = extract_ld_breadcrumb(soup)
    content_hash = stable_sha1(normalize_for_hash(raw_text))
    article_id = stable_sha1(article_url)
    normalized_category = category.strip()

    return {
        "article_id": article_id,
        "url": article_url,
        "categories": [normalized_category] if normalized_category else [],
        "breadcrumb": breadcrumb,
        "title": title,
        "content_hash": content_hash,
        "raw_text": raw_text,
        "blocks": blocks,
    }


def deduplicate_articles(articles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def merge_categories(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
        merged: set[str] = set()
        for source in (left, right):
            for value in source.get("categories", []):
                text = str(value).strip()
                if text:
                    merged.add(text)
        return sorted(merged)

    def choose_richer(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        left_len = len(str(left.get("raw_text", "")))
        right_len = len(str(right.get("raw_text", "")))

        winner = right if right_len > left_len else left
        loser = left if winner is right else right
        winner["categories"] = merge_categories(left, right)
        winner_breadcrumb = winner.get("breadcrumb")
        if not isinstance(winner_breadcrumb, list) or not winner_breadcrumb:
            loser_breadcrumb = loser.get("breadcrumb")
            if isinstance(loser_breadcrumb, list) and loser_breadcrumb:
                winner["breadcrumb"] = loser_breadcrumb
        return winner

    by_url: dict[str, dict[str, Any]] = {}
    duplicate_url = 0
    for article in articles:
        normalized_url = str(article.get("url") or "").strip()
        if not normalized_url:
            continue

        existing = by_url.get(normalized_url)
        if existing is None:
            by_url[normalized_url] = article
            continue

        duplicate_url += 1
        by_url[normalized_url] = choose_richer(existing, article)

    by_hash: dict[str, dict[str, Any]] = {}
    duplicate_hash = 0
    for article in by_url.values():
        content_hash = str(article.get("content_hash") or "").strip()
        if not content_hash:
            content_hash = stable_sha1(normalize_for_hash(str(article.get("raw_text", ""))))

        existing = by_hash.get(content_hash)
        if existing is None:
            by_hash[content_hash] = article
            continue

        duplicate_hash += 1
        by_hash[content_hash] = choose_richer(existing, article)

    deduped = list(by_hash.values())
    stats = {
        "duplicate_url": duplicate_url,
        "duplicate_content_hash": duplicate_hash,
        "after_url_dedup": len(by_url),
        "after_dedup": len(deduped),
        "before_dedup": len(articles),
    }
    return deduped, stats


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_raw_records(path: Path, progress_every: int = RAW_LOAD_PROGRESS_EVERY) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        print(f"[load_raw_records] Input file not found: {path}", flush=True)
        return [], 0

    records: list[dict[str, Any]] = []
    invalid_lines = 0
    total_lines = 0
    start_time = time.perf_counter()

    print(f"[load_raw_records] Reading: {path}", flush=True)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                invalid_lines += 1

            if progress_every > 0 and total_lines % progress_every == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[load_raw_records] lines={total_lines:,} valid={len(records):,} "
                    f"invalid={invalid_lines:,} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_time
    print(
        f"[load_raw_records] Done. lines={total_lines:,} valid={len(records):,} "
        f"invalid={invalid_lines:,} elapsed={elapsed:.1f}s",
        flush=True,
    )

    return records, invalid_lines


def parse_raw_records(
    raw_records: list[dict[str, Any]],
    progress_every: int = PARSE_PROGRESS_EVERY,
) -> tuple[list[dict[str, Any]], int]:
    articles: list[dict[str, Any]] = []
    parse_failed = 0
    total = len(raw_records)
    start_time = time.perf_counter()

    print(f"[parse_raw_records] Start parsing {total:,} records", flush=True)

    for idx, record in enumerate(raw_records, start=1):
        html = record.get("html")
        url = str(record.get("url") or "").strip()
        category = str(record.get("category") or "").strip()

        if not isinstance(html, str) or not html.strip() or not url:
            parse_failed += 1
            continue

        article = parse_article(html, url, category)
        if article is None:
            parse_failed += 1
            continue

        articles.append(article)

        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            elapsed = time.perf_counter() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[parse_raw_records] processed={idx:,}/{total:,} ok={len(articles):,} "
                f"failed={parse_failed:,} rate={rate:.2f} rec/s elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    print(
        f"[parse_raw_records] Done. ok={len(articles):,} failed={parse_failed:,} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )

    return articles, parse_failed


def main() -> None:
    overall_start = time.perf_counter()
    print("[main] Parse pipeline started", flush=True)
    print(f"[main] Input: {RAW_HTML_INPUT_PATH}", flush=True)
    print(f"[main] Output structured: {STRUCTURED_OUTPUT_PATH}", flush=True)
    print(f"[main] Output report: {PARSE_REPORT_OUTPUT_PATH}", flush=True)

    ensure_output_dirs()
    print("[main] Output directory ensured", flush=True)

    started_at = utc_now_iso()
    raw_records, invalid_raw_lines = load_raw_records(RAW_HTML_INPUT_PATH)
    articles, parse_failed = parse_raw_records(raw_records)

    print("[main] Running article-level dedup", flush=True)
    deduped_articles, dedup_stats = deduplicate_articles(articles)
    print(
        f"[main] Dedup done. before={len(articles):,} after={len(deduped_articles):,} "
        f"duplicate_url={dedup_stats.get('duplicate_url', 0):,} "
        f"duplicate_hash={dedup_stats.get('duplicate_content_hash', 0):,}",
        flush=True,
    )
    print("[main] Writing structured JSONL", flush=True)
    write_jsonl(STRUCTURED_OUTPUT_PATH, deduped_articles)

    report = {
        "parse_started_at": started_at,
        "parse_finished_at": utc_now_iso(),
        "raw_input_path": str(RAW_HTML_INPUT_PATH),
        "structured_output_path": str(STRUCTURED_OUTPUT_PATH),
        "raw_records": len(raw_records),
        "invalid_raw_lines": invalid_raw_lines,
        "parsed_articles_before_dedup": len(articles),
        "parsed_articles_after_dedup": len(deduped_articles),
        "parse_failed_records": parse_failed,
        "article_dedup": dedup_stats,
    }
    print("[main] Writing parse report JSON", flush=True)
    write_json(PARSE_REPORT_OUTPUT_PATH, report)

    elapsed_total = time.perf_counter() - overall_start
    print("\n=== Parse Completed ===")
    print(f"Raw input: {len(raw_records)} -> {RAW_HTML_INPUT_PATH}")
    print(f"Structured articles saved: {len(deduped_articles)} -> {STRUCTURED_OUTPUT_PATH}")
    print(f"Parse failed records: {parse_failed}")
    print(f"Parse report: {PARSE_REPORT_OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed_total:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
