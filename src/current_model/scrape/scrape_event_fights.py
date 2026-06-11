from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.db.schema import initialize_database
from src.current_model.utils.http import build_session, fetch_html
from src.current_model.utils.ids import fighter_id_from_url, fight_id_from_url
from src.current_model.utils.logging import append_error_row, configure_logging
from src.current_model.utils.storage import load_csv, save_csv


BASE_URL = "http://ufcstats.com"
FIGHT_COLUMNS = [
    "fight_id",
    "fight_url",
    "event_id",
    "event_name",
    "event_date_raw",
    "fighter_1_name",
    "fighter_1_url",
    "fighter_1_id",
    "fighter_2_name",
    "fighter_2_url",
    "fighter_2_id",
    "winner_name",
    "winner_id",
    "weight_class",
    "method_raw",
    "method_details",
    "ending_round",
    "ending_time_raw",
    "scheduled_rounds",
    "referee",
    "scraped_at",
]


def first_match(text: str | None, patterns: list[str]) -> str | None:
    if text is None:
        return None
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_label_map(soup: BeautifulSoup) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for tag in soup.find_all(["li", "p", "div", "span"]):
        text = tag.get_text(" ", strip=True)
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        label = label.strip().lower()
        value = value.strip()
        if label and value and label not in label_map:
            label_map[label] = value
    return label_map


def parse_fight_page(html: str, event_row: pd.Series, source_url: str | None = None) -> dict[str, str | int | None]:
    soup = BeautifulSoup(html, "html.parser")
    fight_link = soup.select_one('link[rel="canonical"]')
    fight_url = source_url or (fight_link.get("href") if fight_link else None)
    fight_id = fight_id_from_url(fight_url)

    fighter_blocks = soup.select('div.b-fight-details__person')
    fighters: list[dict[str, str | None]] = []
    for block in fighter_blocks:
        link = block.select_one('a[href*="fighter-details/"]')
        if not link:
            continue
        fighter_url = link.get("href")
        fighter_id = fighter_id_from_url(fighter_url)
        fighter_name = link.get_text(" ", strip=True)
        status = block.select_one(".b-fight-details__person-status")
        result = status.get_text(" ", strip=True) if status else None
        fighters.append({"name": fighter_name, "url": fighter_url, "id": fighter_id, "result": result})
    fighters = fighters[:2]

    winner_name = None
    winner_id = None
    for fighter in fighters:
        if fighter.get("result") == "W":
            winner_name = fighter.get("name")
            winner_id = fighter.get("id")
            break

    labels = extract_label_map(soup)

    detail_content = soup.select_one(".b-fight-details__content")
    detail_parts: dict[str, str] = {}
    if detail_content:
        for item in detail_content.select(".b-fight-details__text-item, .b-fight-details__text-item_first"):
            label_tag = item.select_one(".b-fight-details__label")
            if label_tag is None:
                continue
            label = label_tag.get_text(" ", strip=True).rstrip(":").strip().lower()
            text_parts = list(item.stripped_strings)
            if not text_parts:
                continue
            if text_parts[0].lower().startswith(label):
                text_parts = text_parts[1:]
            value = " ".join(text_parts).strip()
            if value:
                detail_parts[label] = value

    round_text = detail_parts.get("round") or labels.get("round") or labels.get("ending round")
    ending_round = int(round_text) if round_text and str(round_text).isdigit() else None
    time_text = detail_parts.get("time") or labels.get("time") or labels.get("ending time")
    time_format = detail_parts.get("time format") or labels.get("time format") or labels.get("round format") or labels.get("format")
    scheduled_rounds = first_match(time_format, [r"(\d+)\s*rnd", r"(\d+)"])
    scheduled_rounds_int = int(scheduled_rounds) if scheduled_rounds and scheduled_rounds.isdigit() else None
    method_raw = detail_parts.get("method") or labels.get("method")
    referee = detail_parts.get("referee") or labels.get("referee")
    method_details = detail_parts.get("details") or labels.get("details")
    fight_title = soup.select_one(".b-fight-details__fight-title")
    weight_class = None
    if fight_title:
        title_text = fight_title.get_text(" ", strip=True)
        weight_class = re.sub(r"\s*Bout$", "", title_text, flags=re.IGNORECASE).strip() or None

    return {
        "fight_id": fight_id,
        "fight_url": fight_url,
        "event_id": event_row.get("event_id"),
        "event_name": event_row.get("event_name"),
        "event_date_raw": event_row.get("event_date_raw"),
        "fighter_1_name": fighters[0]["name"] if len(fighters) > 0 else None,
        "fighter_1_url": fighters[0]["url"] if len(fighters) > 0 else None,
        "fighter_1_id": fighters[0]["id"] if len(fighters) > 0 else None,
        "fighter_2_name": fighters[1]["name"] if len(fighters) > 1 else None,
        "fighter_2_url": fighters[1]["url"] if len(fighters) > 1 else None,
        "fighter_2_id": fighters[1]["id"] if len(fighters) > 1 else None,
        "winner_name": winner_name,
        "winner_id": winner_id,
        "weight_class": weight_class or labels.get("weight class") or labels.get("weightclass"),
        "method_raw": method_raw,
        "method_details": method_details,
        "ending_round": ending_round,
        "ending_time_raw": time_text,
        "scheduled_rounds": scheduled_rounds_int,
        "referee": referee,
    }


def collect_fight_urls(event_html: str) -> list[str]:
    soup = BeautifulSoup(event_html, "html.parser")
    urls: list[str] = []
    for row in soup.select("tr.js-fight-details-click, tr[data-link*='fight-details/']"):
        href = row.get("data-link")
        if not href:
            onclick = row.get("onclick") or ""
            match = re.search(r"https?://[^'\"]*fight-details/[^'\"]+", onclick)
            href = match.group(0) if match else None
        if href and href not in urls:
            urls.append(href)
    return urls


def main(force: bool = False) -> pd.DataFrame:
    logger = configure_logging()
    session = build_session()
    data_dir = Path("data") / "current_model"
    raw_dir = data_dir / "raw"
    html_dir = data_dir / "raw_html"
    db_path = Path("ufc_stats.db")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(str(db_path))

    events = load_csv(raw_dir / "events_raw.csv")
    if events.empty:
        empty = pd.DataFrame(columns=FIGHT_COLUMNS)
        save_csv(empty, raw_dir / "fights_raw.csv")
        with sqlite3.connect(str(db_path)) as conn:
            empty.to_sql("fights_raw", conn, if_exists="replace", index=False)
        logger.warning("No events available to scrape fights from")
        return empty

    existing = load_csv(raw_dir / "fights_raw.csv")
    existing_ids = set(existing["fight_id"].astype(str)) if not existing.empty and "fight_id" in existing else set()

    rows: list[dict[str, object]] = []
    for _, event_row in tqdm(events.iterrows(), total=len(events), desc="events"):
        event_id = event_row["event_id"]
        event_html = fetch_html(
            event_row["event_url"],
            html_dir / "events" / f"{event_id}.html",
            force=force,
            session=session,
            logger=logger,
        )
        if "Checking your browser" in event_html:
            append_error_row(
                Path("logs") / "errors.csv",
                {
                    "stage": "event_fights",
                    "entity_id": event_id,
                    "url": event_row["event_url"],
                    "message": "Blocked by browser challenge",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue

        for fight_url in collect_fight_urls(event_html):
            fight_id = fight_id_from_url(fight_url)
            if not fight_id or (not force and fight_id in existing_ids):
                continue
            fight_html = fetch_html(
                fight_url,
                html_dir / "fights" / f"{fight_id}.html",
                force=force,
                session=session,
                logger=logger,
            )
            try:
                rows.append(parse_fight_page(fight_html, event_row, fight_url))
            except Exception as exc:
                append_error_row(
                    Path("logs") / "errors.csv",
                    {
                        "stage": "fight_metadata",
                        "entity_id": fight_id,
                        "url": fight_url,
                        "message": str(exc),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

    scraped_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["scraped_at"] = scraped_at

    new_frame = pd.DataFrame(rows, columns=FIGHT_COLUMNS)
    if not existing.empty and not force:
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["fight_id"], keep="last")
    if combined.empty:
        combined = pd.DataFrame(columns=FIGHT_COLUMNS)

    save_csv(combined, raw_dir / "fights_raw.csv")
    with sqlite3.connect(str(db_path)) as conn:
        combined.to_sql("fights_raw", conn, if_exists="replace", index=False)
    logger.info("Saved %s fights", len(combined))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape UFCStats fights from completed events.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
