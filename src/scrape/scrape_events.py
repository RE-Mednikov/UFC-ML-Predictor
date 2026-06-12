from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clean.parsers import parse_date_to_iso
from src.db.schema import initialize_database
from src.utils.http import build_session, fetch_html
from src.utils.ids import event_id_from_url
from src.utils.logging import append_error_row, configure_logging
from src.utils.storage import load_csv, save_csv


BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
EVENT_COLUMNS = ["event_id", "event_url", "event_name", "event_date_raw", "event_location_raw", "scraped_at"]


def parse_events(html: str) -> list[dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str | None]] = []
    for row in soup.select("table.b-statistics__table-events tbody tr"):
        anchor = row.select_one('a[href*="event-details/"]')
        if anchor is None:
            continue
        event_url = anchor.get("href")
        event_id = event_id_from_url(event_url)
        if not event_id:
            continue
        event_name = anchor.get_text(" ", strip=True)
        date_tag = row.select_one(".b-statistics__date")
        location_cell = row.select_one("td.b-statistics__table-col_style_big-top-padding")
        event_date_raw = date_tag.get_text(" ", strip=True) if date_tag else None
        event_location_raw = location_cell.get_text(" ", strip=True) if location_cell else None
        rows.append(
            {
                "event_id": event_id,
                "event_url": event_url,
                "event_name": event_name,
                "event_date_raw": event_date_raw,
                "event_location_raw": event_location_raw,
            }
        )
    unique_rows = {row["event_id"]: row for row in rows}
    return list(unique_rows.values())


def main(limit: int | None = 2, force: bool = False) -> pd.DataFrame:
    logger = configure_logging()
    session = build_session()
    raw_dir = Path("data") / "raw"
    html_dir = Path("data") / "raw_html" / "events"
    db_path = Path("ufc_stats.db")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(str(db_path))

    html = fetch_html(EVENTS_URL, html_dir / "completed_events.html", force=force, session=session, logger=logger)
    if "Checking your browser" in html or "Loading…" in html:
        append_error_row(
            Path("logs") / "errors.csv",
            {
                "stage": "events",
                "entity_id": None,
                "url": EVENTS_URL,
                "message": "Blocked by browser challenge",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    events = parse_events(html)
    today_iso = datetime.now().date().isoformat()
    events = [event for event in events if (parse_date_to_iso(event.get("event_date_raw")) or today_iso) <= today_iso]
    if limit is not None:
        events = events[:limit]

    existing = load_csv(raw_dir / "events_raw.csv")
    existing_ids = set(existing["event_id"].astype(str)) if not existing.empty and "event_id" in existing else set()
    scraped_at = datetime.now(timezone.utc).isoformat()
    for event in events:
        event["scraped_at"] = scraped_at

    new_rows = [event for event in events if force or event["event_id"] not in existing_ids]
    new_frame = pd.DataFrame(new_rows, columns=EVENT_COLUMNS)
    if not existing.empty and not force:
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    if combined.empty:
        combined = pd.DataFrame(columns=EVENT_COLUMNS)

    save_csv(combined, raw_dir / "events_raw.csv")
    with sqlite3.connect(str(db_path)) as conn:
        combined.to_sql("events_raw", conn, if_exists="replace", index=False)
    logger.info("Saved %s events", len(combined))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape UFCStats completed events.")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, force=args.force)
