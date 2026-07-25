"""Scraper for SHL Individual Test Solutions catalog.

Requirements:
- Scrape only Individual Test Solutions (type=1)
- Handle pagination (start param increments by 12)
- Apply rate limiting and retry logic
- Extract name, url, description, duration, adaptive_support, remote_support, test_type
- Persist cleaned data to JSON and optionally CSV
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import CLEANED_CATALOG_PATH
from utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.shl.com/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
PAGE_SIZE = 12
TYPE_INDIVIDUAL = 1  # SHL uses type=1 for Individual Test Solutions


def _session() -> requests.Session:
    session = requests.Session()
    retry_cfg = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


_GLOBAL_SESSION = _session()


@dataclass
class Assessment:
    name: str
    url: str
    description: str
    duration: Optional[int]
    adaptive_support: str
    remote_support: str
    test_type: List[str]


class ScraperError(Exception):
    """Raised when scraping fails or requirements are not met."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.5, max=2),
    retry=retry_if_exception_type(requests.RequestException),
)
def _fetch(url: str) -> str:
    logger.debug(f"Fetching URL: {url}")
    resp = _GLOBAL_SESSION.get(url, timeout=(8, 10), allow_redirects=True)
    resp.raise_for_status()
    time.sleep(0.1)  # light rate limit
    return resp.text


def _parse_table(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    table = None
    for tbl in soup.find_all("table"):
        header = tbl.find("tr")
        if not header:
            continue
        first_cell = header.find("th") or header.find("td")
        if first_cell and "Individual Test Solutions" in first_cell.get_text(strip=True):
            table = tbl
            break
    if table is None:
        raise ScraperError("Individual Test Solutions table not found")

    items: List[Dict[str, str]] = []
    for row in table.find_all("tr"):
        cols = row.find_all(["td", "th"])
        if len(cols) < 4 or "Individual Test Solutions" in cols[0].get_text():
            continue
        name_cell = cols[0]
        name = name_cell.get_text(strip=True)
        link = name_cell.find("a")
        href = link["href"] if link and link.has_attr("href") else ""
        url = href
        if href.startswith("/"):
            url = "https://www.shl.com" + href
        remote = "Yes" if cols[1].get_text(strip=True) else "No"
        adaptive = "Yes" if cols[2].get_text(strip=True) else "No"
        test_type = [t for t in re.split(r"\s+", cols[3].get_text(strip=True)) if t]
        items.append(
            {
                "name": name,
                "url": url,
                "remote_support": remote,
                "adaptive_support": adaptive,
                "test_type": test_type,
            }
        )
    return items


def _extract_description_and_duration(url: str) -> tuple[str, Optional[int]]:
    if not url:
        return "", None
    try:
        html = _fetch(url)
    except BaseException as exc:  # noqa: BLE001
        logger.warning(f"Failed to fetch detail page {url}: {exc}")
        return "", None

    soup = BeautifulSoup(html, "lxml")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"].strip() if meta_desc and meta_desc.has_attr("content") else ""

    duration = None
    duration_text = None
    for pattern in [r"(\d+)\s*minutes", r"(\d+)\s*mins", r"Duration[:\s]+(\d+)"]:
        match = re.search(pattern, soup.get_text(" ", strip=True), flags=re.IGNORECASE)
        if match:
            duration_text = match.group(1)
            break
    if duration_text:
        try:
            duration = int(duration_text)
        except ValueError:
            duration = None

    return description, duration


def _fill_missing_details(items: List[Assessment], attempts: int = 2) -> None:
    """Retry fetching descriptions/durations for items still missing details."""
    for attempt in range(1, attempts + 1):
        missing = [a for a in items if not a.description]
        if not missing:
            break
        logger.info("Detail retry pass %d: attempting %d items", attempt, len(missing))
        for assmt in missing:
            desc, dur = _extract_description_and_duration(assmt.url)
            if desc:
                assmt.description = desc
            if dur is not None:
                assmt.duration = dur


def scrape_individual_tests(max_pages: int = 40, save_csv: bool = True, enrich_details: bool = True) -> List[Assessment]:
    """Scrape the SHL catalog for Individual Test Solutions.

    Args:
        max_pages: safety cap on pagination iterations
        save_csv: whether to emit a CSV alongside JSON for debugging
    """
    logger.info(f"Starting scrape of SHL Individual Test Solutions... (enrich_details={enrich_details})")
    all_items: List[Assessment] = []

    starts = [page * PAGE_SIZE for page in range(max_pages)]

    def process(starts_batch):
        new_items = []
        failed = []
        for start in starts_batch:
            page_no = start // PAGE_SIZE + 1
            url = f"{BASE_URL}?start={start}&type={TYPE_INDIVIDUAL}"
            try:
                html = _fetch(url)
            except BaseException as exc:  # noqa: BLE001
                logger.warning(f"Failed to fetch page {page_no} ({url}): {exc}")
                failed.append(start)
                continue
            page_items = _parse_table(html)
            if not page_items:
                logger.info(f"Page {page_no} returned no items; continuing")
                failed.append(start)
                continue
            logger.info(f"Parsed {len(page_items)} items from page {page_no}")

            for item in page_items:
                if enrich_details:
                    description, duration = _extract_description_and_duration(item["url"])
                else:
                    description, duration = "", None
                assessment = Assessment(
                    name=item["name"],
                    url=item["url"],
                    description=description,
                    duration=duration,
                    adaptive_support=item["adaptive_support"],
                    remote_support=item["remote_support"],
                    test_type=item["test_type"],
                )
                new_items.append(assessment)
        return new_items, failed

    failed_starts = starts
    for attempt in range(6):
        if not failed_starts:
            break
        logger.info(f"Pass {attempt+1}: processing {len(failed_starts)} pages")
        new_items, failed_starts = process(failed_starts)
        all_items.extend(new_items)
        # deduplicate by URL
        seen = {}
        for itm in all_items:
            seen[itm.url] = itm
        all_items = list(seen.values())
        logger.info(f"Accumulated {len(all_items)} items; remaining failed pages: {failed_starts}")

    if enrich_details:
        _fill_missing_details(all_items, attempts=2)

    if len(all_items) < 377:
        raise ScraperError(f"Scraped {len(all_items)} items; expected at least 377 Individual Test Solutions")

    data = [asdict(a) for a in all_items]
    CLEANED_CATALOG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(all_items)} items to {CLEANED_CATALOG_PATH}")

    if save_csv:
        csv_path = CLEANED_CATALOG_PATH.with_suffix(".csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        logger.info(f"Saved CSV copy to {csv_path}")

    return all_items


if __name__ == "__main__":  # pragma: no cover
    scrape_individual_tests()
