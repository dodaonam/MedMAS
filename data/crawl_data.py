import aiohttp
import asyncio
import functools
import inspect
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from data.parse_data import extract_article_links, normalize_url

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

RAW_HTML_OUTPUT_PATH = RAW_DIR / "vinmec_html.jsonl"
REPORT_OUTPUT_PATH = RAW_DIR / "vinmec_crawl_report.json"
ERROR_LOG_PATH = RAW_DIR / "vinmec_crawl_errors.jsonl"

MAX_CONCURRENCY = 5
REQUEST_TIMEOUT_SECONDS = 40
MAX_RETRIES = 4
CATEGORY_PAGE_DELAY_SECONDS = (0.8, 1.5)
REQUEST_DELAY_SECONDS = (0.2, 0.8)
MAX_PAGES_PER_CATEGORY = 500
WRITE_FLUSH_EVERY = 50
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

CATEGORY_SOURCES = [
    ("https://www.vinmec.com/vie/chan-thuong-chinh-hinh-y-hoc-the-thao/page_0", "Chấn thương chỉnh hình - Y khoa"),
    ("https://www.vinmec.com/vie/trung-tam-cong-nghe-cao/page_0", "Công nghệ cao"),
    ("https://www.vinmec.com/vie/trung-tam-suc-khoe-phu-nu/page_0", "Sức khỏe phụ nữ"),
    ("https://www.vinmec.com/vie/trung-tam-nhi/page_0", "Nhi khoa"),
    ("https://www.vinmec.com/vie/trung-tam-vu/page_0", "Vú"),
    ("https://www.vinmec.com/vie/suc-khoe-tong-quat/page_0", "Sức khỏe tổng quát"),
    ("https://www.vinmec.com/vie/than-kinh/page_0", "Thần kinh"),
    ("https://www.vinmec.com/vie/tieu-hoa-gan-mat/page_0", "Tiêu hóa gan mật"),
    ("https://www.vinmec.com/vie/tim-mach/page_0", "Tim mạch"),
    ("https://www.vinmec.com/vie/thong-tin-suc-khoe/page_0", "Thông tin sức khỏe"),
    ("https://www.vinmec.com/vie/trung-tam-cham-soc-suc-khoe-tinh-than/page_0", "Sức khỏe tinh thần"),
    ("https://www.vinmec.com/vie/trung-tam-mat/page_0", "Mắt"),
    ("https://www.vinmec.com/vie/ung-buou/page_0", "Ung bướu"),
    ("https://www.vinmec.com/vie/trung-tam-vac-xin/page_0", "Vắc xin"),
    ("https://www.vinmec.com/vie/te-bao-goc-va-cong-nghe-gen/page_0", "Tế bào gốc và công nghệ gen"),
    ("https://www.vinmec.com/vie/y-hoc-co-truyen/page_0", "Y học cổ truyền"),
    ("https://www.vinmec.com/vie/tham-my/page_0", "Thẩm mỹ"),
    ("https://www.vinmec.com/vie/dich-covid-19/page_0", "Covid-19"),
    ("https://www.vinmec.com/vie/dinh-duong/page_0", "Dinh dưỡng"),
    ("https://www.vinmec.com/vie/song-khoe/page_0", "Sống khỏe"),
    ("https://www.vinmec.com/vie/mien-dich-di-ung/page_0", "Miễn dịch dị ứng"),
]


@dataclass(slots=True)
class CrawlSettings:
    max_concurrency: int = MAX_CONCURRENCY
    request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES


@dataclass(slots=True)
class FetchResult:
    url: str
    status_code: int | None
    html: str | None
    error: str | None
    fetched_at: str


def timing_decorator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            f"Function {func.__name__} took {elapsed_time:.2f} seconds "
            f"({str(timedelta(seconds=int(elapsed_time)))})"
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            f"Function {func.__name__} took {elapsed_time:.2f} seconds "
            f"({str(timedelta(seconds=int(elapsed_time)))})"
        )
        return result

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VinmecCrawler:
    def __init__(self, category_sources: list[tuple[str, str]], settings: CrawlSettings):
        self.category_sources = category_sources
        self.settings = settings

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:126.0) Gecko/20100101 Firefox/126.0",
        ]
        self.user_agent = random.choice(self.user_agents)
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.7,en;q=0.5",
            "Connection": "keep-alive",
        }

        self.semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        self.robots_parser: RobotFileParser | None = None
        self.robots_loaded = False
        self.io_lock = asyncio.Lock()
        self.raw_html_count = 0
        self.error_count = 0
        self.output_streams: dict[Path, Any] = {}
        self.pending_writes: dict[Path, int] = {}

        self.fetched_article_urls: set[str] = set()

    async def _append_jsonl_record(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        async with self.io_lock:
            handle = self.output_streams.get(path)
            if handle is None:
                handle = path.open("a", encoding="utf-8")
                self.output_streams[path] = handle
                self.pending_writes[path] = 0

            handle.write(line)
            pending = self.pending_writes.get(path, 0) + 1
            self.pending_writes[path] = pending
            if pending >= WRITE_FLUSH_EVERY:
                handle.flush()
                self.pending_writes[path] = 0

    async def _open_output_streams(self) -> None:
        async with self.io_lock:
            for path in (RAW_HTML_OUTPUT_PATH, ERROR_LOG_PATH):
                path.parent.mkdir(parents=True, exist_ok=True)
                self.output_streams[path] = path.open("a", encoding="utf-8")
                self.pending_writes[path] = 0

    async def _close_output_streams(self) -> None:
        async with self.io_lock:
            for path, handle in self.output_streams.items():
                try:
                    handle.flush()
                finally:
                    handle.close()
                self.pending_writes[path] = 0
            self.output_streams.clear()

    async def _log_error(self, *, url: str, category: str, error: str, status_code: int | None = None) -> None:
        payload = {
            "url": url,
            "category": category,
            "status_code": status_code,
            "error": error,
            "logged_at": utc_now_iso(),
        }
        self.error_count += 1
        await self._append_jsonl_record(ERROR_LOG_PATH, payload)

    def _init_robots(self) -> None:
        if self.robots_loaded:
            return

        sample_url = self.category_sources[0][0]
        parsed = urlsplit(sample_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            self.robots_parser = parser
            print(f"Loaded robots.txt: {robots_url}")
        except Exception as exc:
            self.robots_parser = None
            print(f"Warning: could not read robots.txt ({robots_url}): {exc}")
        finally:
            self.robots_loaded = True

    def can_fetch(self, url: str) -> bool:
        if self.robots_parser is None:
            return True
        try:
            return self.robots_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def fetch_url(self, session: aiohttp.ClientSession, url: str, *, category: str) -> FetchResult:
        if not self.can_fetch(url):
            return FetchResult(
                url=url,
                status_code=None,
                html=None,
                error="disallowed_by_robots",
                fetched_at=utc_now_iso(),
            )

        last_status: int | None = None
        last_error: str | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            await asyncio.sleep(random.uniform(*REQUEST_DELAY_SECONDS))
            try:
                async with self.semaphore:
                    async with session.get(url, headers=self.headers, timeout=self.settings.request_timeout_seconds) as response:
                        status = response.status
                        fetched_at = utc_now_iso()
                        if status == 200:
                            html = await response.text()
                            return FetchResult(
                                url=url,
                                status_code=status,
                                html=html,
                                error=None,
                                fetched_at=fetched_at,
                            )

                        body_preview = (await response.text())[:180]
                        if status not in RETRYABLE_STATUS_CODES:
                            return FetchResult(
                                url=url,
                                status_code=status,
                                html=None,
                                error=f"http_{status}: {body_preview.strip()}",
                                fetched_at=fetched_at,
                            )
                        last_status = status
                        last_error = f"http_{status}: {body_preview.strip()}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.settings.max_retries:
                backoff = min(2 ** (attempt - 1), 10) + random.uniform(0.0, 0.6)
                await asyncio.sleep(backoff)

        return FetchResult(
            url=url,
            status_code=last_status,
            html=None,
            error=last_error or "unknown_fetch_error",
            fetched_at=utc_now_iso(),
        )

    async def scrape_article(self, session: aiohttp.ClientSession, article_url: str, category: str) -> bool:
        fetch_result = await self.fetch_url(session, article_url, category=category)
        if fetch_result.html is None:
            await self._log_error(
                url=article_url,
                category=category,
                status_code=fetch_result.status_code,
                error=fetch_result.error or "fetch_failed",
            )
            return False

        normalized_url = normalize_url(article_url)
        self.fetched_article_urls.add(normalized_url)

        raw_record = {
            "url": article_url,
            "normalized_url": normalized_url,
            "status_code": fetch_result.status_code,
            "html": fetch_result.html,
            "fetched_at": fetch_result.fetched_at,
            "category": category,
            "source": "vinmec",
        }
        await self._append_jsonl_record(RAW_HTML_OUTPUT_PATH, raw_record)
        self.raw_html_count += 1
        return True

    @timing_decorator
    async def crawl(self) -> dict[str, Any]:
        self._init_robots()
        await self._open_output_streams()

        try:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for category_base_url, category_tag in self.category_sources:
                    print(f"\\n=== Category: {category_tag} ===")
                    page_number = 0
                    pages_without_new_links = 0

                    while page_number < MAX_PAGES_PER_CATEGORY:
                        page_url = category_base_url.replace("page_0", f"page_{page_number}")
                        page_result = await self.fetch_url(session, page_url, category=category_tag)
                        if page_result.html is None:
                            await self._log_error(
                                url=page_url,
                                category=category_tag,
                                status_code=page_result.status_code,
                                error=page_result.error or "category_page_fetch_failed",
                            )
                            print(f"Stop category {category_tag}: page fetch failed at {page_number}")
                            break

                        links = extract_article_links(page_result.html, category_base_url)
                        if not links:
                            print(f"Stop category {category_tag}: no article links on page {page_number}")
                            break

                        new_links = [link for link in links if normalize_url(link) not in self.fetched_article_urls]
                        if not new_links:
                            pages_without_new_links += 1
                            if pages_without_new_links >= 2:
                                print(f"Stop category {category_tag}: repeated pages with no new links")
                                break
                        else:
                            pages_without_new_links = 0

                        print(
                            f"Page {page_number} | links: {len(links)} | new: {len(new_links)} | "
                            f"total_fetched: {len(self.fetched_article_urls)}"
                        )

                        tasks = [self.scrape_article(session, link, category_tag) for link in new_links]
                        results = await asyncio.gather(*tasks)

                        ok_count = sum(1 for ok in results if ok)
                        print(f"Fetched {ok_count}/{len(new_links)} new raw articles on page {page_number}")

                        page_number += 1
                        await asyncio.sleep(random.uniform(*CATEGORY_PAGE_DELAY_SECONDS))
        finally:
            await self._close_output_streams()

        return {
            "crawl_started_at": None,
            "crawl_finished_at": utc_now_iso(),
            "source": "vinmec",
            "raw_html_records": self.raw_html_count,
            "fetched_article_urls": len(self.fetched_article_urls),
            "errors": self.error_count,
            "request_config": {
                "max_concurrency": self.settings.max_concurrency,
                "timeout_seconds": self.settings.request_timeout_seconds,
                "max_retries": self.settings.max_retries,
            },
            "paging_config": {
                "max_pages_per_category": MAX_PAGES_PER_CATEGORY,
                "category_delay_seconds": list(CATEGORY_PAGE_DELAY_SECONDS),
                "request_delay_seconds": list(REQUEST_DELAY_SECONDS),
            },
        }


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def reset_output_files() -> None:
    for path in (RAW_HTML_OUTPUT_PATH, ERROR_LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


@timing_decorator
async def main() -> None:
    ensure_output_dirs()
    reset_output_files()

    started_at = utc_now_iso()
    wall_start = time.time()

    crawler = VinmecCrawler(CATEGORY_SOURCES, CrawlSettings())
    report = await crawler.crawl()

    report["crawl_started_at"] = started_at
    report["crawl_finished_at"] = utc_now_iso()
    report["elapsed_seconds"] = round(time.time() - wall_start, 2)

    write_json(REPORT_OUTPUT_PATH, report)

    print("\\n=== Crawl Completed ===")
    print(f"Raw HTML saved: {report['raw_html_records']} -> {RAW_HTML_OUTPUT_PATH}")
    print(f"Errors logged: {report['errors']} -> {ERROR_LOG_PATH}")
    print(f"Crawl report: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
