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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clean.parsers import parse_date_to_iso, parse_float, parse_name_parts, parse_percentage, parse_record
from src.db.schema import initialize_database
from src.utils.http import build_session, fetch_html
from src.utils.ids import fighter_id_from_url
from src.utils.logging import append_error_row, configure_logging
from src.utils.storage import load_csv, save_csv


FIGHTER_COLUMNS = [
    "fighter_id",
    "fighter_url",
    "first_name",
    "last_name",
    "full_name",
    "nickname",
    "height_raw",
    "weight_raw",
    "reach_raw",
    "stance",
    "dob_raw",
    "record_raw",
    "profile_SLpM",
    "profile_str_acc",
    "profile_SApM",
    "profile_str_def",
    "profile_TD_avg",
    "profile_TD_acc",
    "profile_TD_def",
    "profile_sub_avg",
    "scraped_at",
]


def find_label_value(text: str | None, label: str) -> str | None:
    if text is None:
        return None
    pattern = rf"{re.escape(label)}\s*:\s*([^\n\r|]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_profile(html: str, fighter_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text("\n", strip=True)
    fighter_id = fighter_id_from_url(fighter_url)

    title = soup.select_one("h2") or soup.select_one("h1")
    full_name = title.get_text(" ", strip=True) if title else fighter_id
    nickname = None
    if full_name and '"' in full_name:
        parts = full_name.split('"')
        if len(parts) >= 3:
            nickname = parts[1].strip()
            full_name = parts[0].replace("(", "").replace(")", "").strip()

    first_name, last_name = parse_name_parts(full_name)

    profile = {
        "fighter_id": fighter_id,
        "fighter_url": fighter_url,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "nickname": nickname,
        "height_raw": find_label_value(full_text, "Height"),
        "weight_raw": find_label_value(full_text, "Weight"),
        "reach_raw": find_label_value(full_text, "Reach"),
        "stance": find_label_value(full_text, "STANCE") or find_label_value(full_text, "Stance"),
        "dob_raw": find_label_value(full_text, "DOB") or find_label_value(full_text, "Date of Birth"),
        "record_raw": find_label_value(full_text, "Record"),
        "profile_SLpM": parse_float(find_label_value(full_text, "SLpM")),
        "profile_str_acc": parse_percentage(find_label_value(full_text, "Str. Acc.") or find_label_value(full_text, "Str Acc")),
        "profile_SApM": parse_float(find_label_value(full_text, "SApM")),
        "profile_str_def": parse_percentage(find_label_value(full_text, "Str. Def.") or find_label_value(full_text, "Str Def")),
        "profile_TD_avg": parse_float(find_label_value(full_text, "TD Avg.")),
        "profile_TD_acc": parse_percentage(find_label_value(full_text, "TD Acc.") or find_label_value(full_text, "TD Acc")),
        "profile_TD_def": parse_percentage(find_label_value(full_text, "TD Def.") or find_label_value(full_text, "TD Def")),
        "profile_sub_avg": parse_float(find_label_value(full_text, "Sub. Avg.")),
    }
    return profile


def main(force: bool = False) -> pd.DataFrame:
    logger = configure_logging()
    session = build_session()
    raw_dir = Path("data") / "raw"
    html_dir = Path("data") / "raw_html" / "fighters"
    db_path = Path("ufc_stats.db")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(str(db_path))

    fights = load_csv(raw_dir / "fights_raw.csv")
    if fights.empty:
        empty = pd.DataFrame(columns=FIGHTER_COLUMNS)
        save_csv(empty, raw_dir / "fighters_raw.csv")
        with sqlite3.connect(str(db_path)) as conn:
            empty.to_sql("fighters_raw", conn, if_exists="replace", index=False)
        logger.warning("No fights available to scrape fighter profiles from")
        return empty

    fighter_urls: list[dict[str, str]] = []
    for _, row in fights.iterrows():
        for column in ("fighter_1_url", "fighter_2_url"):
            fighter_url = row.get(column)
            fighter_id = fighter_id_from_url(fighter_url)
            if fighter_id and all(existing["fighter_id"] != fighter_id for existing in fighter_urls):
                fighter_urls.append({"fighter_id": fighter_id, "fighter_url": fighter_url})

    existing = load_csv(raw_dir / "fighters_raw.csv")
    existing_ids = set(existing["fighter_id"].astype(str)) if not existing.empty and "fighter_id" in existing else set()

    rows: list[dict[str, object]] = []
    for fighter in tqdm(fighter_urls, desc="fighters"):
        fighter_id = fighter["fighter_id"]
        html = fetch_html(
            fighter["fighter_url"],
            html_dir / f"{fighter_id}.html",
            force=force,
            session=session,
            logger=logger,
        )
        if "Checking your browser" in html:
            append_error_row(
                Path("logs") / "errors.csv",
                {
                    "stage": "fighters",
                    "entity_id": fighter_id,
                    "url": fighter["fighter_url"],
                    "message": "Blocked by browser challenge",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        try:
            rows.append(parse_profile(html, fighter["fighter_url"]))
        except Exception as exc:
            append_error_row(
                Path("logs") / "errors.csv",
                {
                    "stage": "fighter_parse",
                    "entity_id": fighter_id,
                    "url": fighter["fighter_url"],
                    "message": str(exc),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    scraped_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["scraped_at"] = scraped_at

    new_rows = [row for row in rows if force or row["fighter_id"] not in existing_ids]
    new_frame = pd.DataFrame(new_rows, columns=FIGHTER_COLUMNS)
    if not existing.empty and not force:
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["fighter_id"], keep="last")
    if combined.empty:
        combined = pd.DataFrame(columns=FIGHTER_COLUMNS)

    save_csv(combined, raw_dir / "fighters_raw.csv")
    with sqlite3.connect(str(db_path)) as conn:
        combined.to_sql("fighters_raw", conn, if_exists="replace", index=False)
    logger.info("Saved %s fighters", len(combined))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape UFCStats fighter profiles.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
