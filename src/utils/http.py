from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency until installed
    sync_playwright = None
    PlaywrightTimeoutError = TimeoutError


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def is_block_page(html: str) -> bool:
    lowered = html.lower()
    return "checking your browser" in lowered or ("loading" in lowered and "noscript" in lowered)


def fetch_html_with_browser(url: str, timeout: int = 30, logger: logging.Logger | None = None) -> str:
    if sync_playwright is None:
        raise RuntimeError("playwright is not installed")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent=DEFAULT_HEADERS["User-Agent"],
                extra_http_headers={"Accept-Language": DEFAULT_HEADERS["Accept-Language"]},
                viewport={"width": 1440, "height": 1200},
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            except PlaywrightTimeoutError:
                pass
            html = page.content()
            if logger:
                logger.info("Fetched %s via browser", url)
            return html
        finally:
            browser.close()


def fetch_html(
    url: str,
    cache_path: str | Path,
    *,
    force: bool = False,
    timeout: int = 30,
    sleep_seconds: float = 1.0,
    session: requests.Session | None = None,
    logger: logging.Logger | None = None,
    browser_fallback: bool = True,
    fetch_mode: Literal["requests_first", "browser_first"] = "browser_first",
) -> str:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force:
        return cache_file.read_text(encoding="utf-8")

    active_session = session or build_session()
    html: str | None = None

    if sleep_seconds:
        time.sleep(sleep_seconds)

    if fetch_mode == "browser_first":
        try:
            html = fetch_html_with_browser(url, timeout=timeout, logger=logger)
        except Exception as exc:
            if logger:
                logger.warning("Browser fetch failed for %s, falling back to requests: %s", url, exc)

    if html is None:
        response = active_session.get(url, timeout=timeout)
        response.raise_for_status()
        html = response.text

        if browser_fallback and is_block_page(html):
            if logger:
                logger.warning("Block page detected for %s, retrying in browser", url)
            try:
                html = fetch_html_with_browser(url, timeout=timeout, logger=logger)
            except Exception as exc:
                if logger:
                    logger.warning("Browser fetch failed for %s: %s", url, exc)

    cache_file.write_text(html, encoding="utf-8")

    if logger:
        if is_block_page(html):
            logger.warning("Block page detected for %s", url)
        else:
            logger.info("Fetched %s", url)
    return html
