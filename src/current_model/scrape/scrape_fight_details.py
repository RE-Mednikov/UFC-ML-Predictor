from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.clean.parsers import normalize_missing, parse_control_time_to_seconds, parse_int, parse_landed_attempted
from src.current_model.db.schema import initialize_database
from src.current_model.utils.http import build_session, fetch_html
from src.current_model.utils.logging import append_error_row, configure_logging
from src.current_model.utils.storage import load_csv, save_csv


FIGHT_STATS_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date_raw",
    "round",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempted",
    "total_strikes_landed",
    "total_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "submission_attempts",
    "reversals",
    "control_time_raw",
    "head_landed",
    "head_attempted",
    "body_landed",
    "body_attempted",
    "leg_landed",
    "leg_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
    "scraped_at",
]


def round_header_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"round\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_headers(table) -> list[str]:
    head = table.select_one("thead.b-fight-details__table-head_rnd, thead.b-fight-details__table-head")
    if head is None:
        return []
    headers = [cell.get_text(" ", strip=True).lower() for cell in head.select("th")]
    normalized: list[str] = []
    td_percent_seen = 0
    for header in headers:
        normalized_header = header
        if header == "td %":
            td_percent_seen += 1
            normalized_header = "td" if td_percent_seen == 1 else "td %"
        normalized.append(normalized_header)
    return normalized


def extract_table_round_rows(table) -> dict[int, Any]:
    round_rows: dict[int, Any] = {}
    current_round: int | None = None
    for child in table.children:
        name = getattr(child, "name", None)
        if name not in {"thead", "tbody"}:
            continue
        text = child.get_text(" ", strip=True)
        detected_round = round_header_from_text(text)
        if detected_round is not None:
            current_round = detected_round
            continue
        if current_round is None:
            continue
        data_row = child.select_one("tr.b-fight-details__table-row")
        if data_row is not None:
            round_rows[current_round] = data_row
    return round_rows


def extract_dual_values(cells: list[Any], headers: list[str]) -> dict[int, dict[str, str | None]]:
    values_by_fighter: dict[int, dict[str, str | None]] = {0: {}, 1: {}}
    for index, header in enumerate(headers[: len(cells)]):
        cell = cells[index]
        values = [text.get_text(" ", strip=True) for text in cell.find_all("p")]
        if len(values) >= 2:
            values_by_fighter[0][header] = values[0]
            values_by_fighter[1][header] = values[1]
        else:
            value = cell.get_text(" ", strip=True) or None
            values_by_fighter[0][header] = value
            values_by_fighter[1][header] = value
    return values_by_fighter


def build_base_row(event_row: pd.Series, fighter_index: int, round_number: int) -> dict[str, object]:
    fighter_names = (event_row.get("fighter_1_name"), event_row.get("fighter_2_name"))
    fighter_ids = (event_row.get("fighter_1_id"), event_row.get("fighter_2_id"))
    opponent_names = (event_row.get("fighter_2_name"), event_row.get("fighter_1_name"))
    opponent_ids = (event_row.get("fighter_2_id"), event_row.get("fighter_1_id"))
    return {
        "fight_id": event_row["fight_id"],
        "event_id": event_row["event_id"],
        "event_date_raw": event_row["event_date_raw"],
        "round": round_number,
        "fighter_id": fighter_ids[fighter_index],
        "fighter_name": fighter_names[fighter_index],
        "opponent_id": opponent_ids[fighter_index],
        "opponent_name": opponent_names[fighter_index],
    }


def format_seconds_as_clock(total_seconds: int | None) -> str | None:
    if total_seconds is None:
        return None
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def merge_general_stats(row: dict[str, object], stats_map: dict[str, str | None]) -> None:
    sig_landed, sig_attempted = parse_landed_attempted(stats_map.get("sig. str.") or stats_map.get("sig. str") or stats_map.get("sig str."))
    total_landed, total_attempted = parse_landed_attempted(stats_map.get("total str."))
    td_landed, td_attempted = parse_landed_attempted(stats_map.get("td") or stats_map.get("td %"))
    row.update(
        {
            "knockdowns": parse_int(stats_map.get("kd")),
            "sig_strikes_landed": sig_landed,
            "sig_strikes_attempted": sig_attempted,
            "total_strikes_landed": total_landed,
            "total_strikes_attempted": total_attempted,
            "takedowns_landed": td_landed,
            "takedowns_attempted": td_attempted,
            "submission_attempts": parse_int(stats_map.get("sub. att") or stats_map.get("sub att")),
            "reversals": parse_int(stats_map.get("rev.") or stats_map.get("rev")),
            "control_time_raw": normalize_missing(stats_map.get("ctrl")),
        }
    )


def merge_sig_strikes(row: dict[str, object], stats_map: dict[str, str | None]) -> None:
    head_landed, head_attempted = parse_landed_attempted(stats_map.get("head"))
    body_landed, body_attempted = parse_landed_attempted(stats_map.get("body"))
    leg_landed, leg_attempted = parse_landed_attempted(stats_map.get("leg"))
    distance_landed, distance_attempted = parse_landed_attempted(stats_map.get("distance"))
    clinch_landed, clinch_attempted = parse_landed_attempted(stats_map.get("clinch"))
    ground_landed, ground_attempted = parse_landed_attempted(stats_map.get("ground"))
    row.update(
        {
            "head_landed": head_landed,
            "head_attempted": head_attempted,
            "body_landed": body_landed,
            "body_attempted": body_attempted,
            "leg_landed": leg_landed,
            "leg_attempted": leg_attempted,
            "distance_landed": distance_landed,
            "distance_attempted": distance_attempted,
            "clinch_landed": clinch_landed,
            "clinch_attempted": clinch_attempted,
            "ground_landed": ground_landed,
            "ground_attempted": ground_attempted,
        }
    )


def parse_fight_stats(html: str, event_row: pd.Series) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("table.b-fight-details__table.js-fight-table")
    if len(tables) < 2:
        return []

    general_rounds = extract_table_round_rows(tables[0])
    sig_rounds = extract_table_round_rows(tables[1])
    round_numbers = sorted(set(general_rounds) | set(sig_rounds))

    records_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for round_number in round_numbers:
        for fighter_index in (0, 1):
            records_by_key[(round_number, fighter_index)] = build_base_row(event_row, fighter_index, round_number)

        general_row = general_rounds.get(round_number)
        if general_row is not None:
            general_headers = extract_headers(tables[0])
            general_values = extract_dual_values(general_row.find_all("td"), general_headers)
            for fighter_index in (0, 1):
                merge_general_stats(records_by_key[(round_number, fighter_index)], general_values[fighter_index])

        sig_row = sig_rounds.get(round_number)
        if sig_row is not None:
            sig_headers = extract_headers(tables[1])
            sig_values = extract_dual_values(sig_row.find_all("td"), sig_headers)
            for fighter_index in (0, 1):
                merge_sig_strikes(records_by_key[(round_number, fighter_index)], sig_values[fighter_index])

    round_rows = [records_by_key[key] for key in sorted(records_by_key)]
    if not round_rows:
        return []

    aggregated_rows: dict[str, dict[str, object]] = {}
    sum_columns = [
        "knockdowns",
        "sig_strikes_landed",
        "sig_strikes_attempted",
        "total_strikes_landed",
        "total_strikes_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "head_landed",
        "head_attempted",
        "body_landed",
        "body_attempted",
        "leg_landed",
        "leg_attempted",
        "distance_landed",
        "distance_attempted",
        "clinch_landed",
        "clinch_attempted",
        "ground_landed",
        "ground_attempted",
    ]

    for row in round_rows:
        fighter_id = str(row["fighter_id"])
        if fighter_id not in aggregated_rows:
            aggregated_rows[fighter_id] = {
                "fight_id": row["fight_id"],
                "event_id": row["event_id"],
                "event_date_raw": row["event_date_raw"],
                "round": 0,
                "fighter_id": row["fighter_id"],
                "fighter_name": row["fighter_name"],
                "opponent_id": row["opponent_id"],
                "opponent_name": row["opponent_name"],
                "control_time_raw": None,
            }
            for column in sum_columns:
                aggregated_rows[fighter_id][column] = 0

        aggregated_row = aggregated_rows[fighter_id]
        for column in sum_columns:
            value = row.get(column)
            if value is not None:
                aggregated_row[column] = int(aggregated_row[column]) + int(value)

        current_control = parse_control_time_to_seconds(aggregated_row.get("control_time_raw")) or 0
        round_control = parse_control_time_to_seconds(row.get("control_time_raw")) or 0
        aggregated_row["control_time_raw"] = format_seconds_as_clock(current_control + round_control)

    return list(aggregated_rows.values())


def main(force: bool = False) -> pd.DataFrame:
    logger = configure_logging()
    session = build_session()
    data_dir = Path("data") / "current_model"
    raw_dir = data_dir / "raw"
    html_dir = data_dir / "raw_html" / "fights"
    db_path = Path("ufc_stats.db")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(str(db_path))

    fights = load_csv(raw_dir / "fights_raw.csv")
    if fights.empty:
        empty = pd.DataFrame(columns=FIGHT_STATS_COLUMNS)
        save_csv(empty, raw_dir / "fight_stats_raw.csv")
        with sqlite3.connect(str(db_path)) as conn:
            empty.to_sql("fight_stats_raw", conn, if_exists="replace", index=False)
        logger.warning("No fights available to scrape fight details from")
        return empty

    existing = load_csv(raw_dir / "fight_stats_raw.csv")
    existing_keys = set()
    if not existing.empty and {"fight_id", "fighter_id"}.issubset(existing.columns):
        existing_keys = set(zip(existing["fight_id"].astype(str), existing["fighter_id"].astype(str)))

    rows: list[dict[str, object]] = []
    for _, fight_row in tqdm(fights.iterrows(), total=len(fights), desc="fights"):
        fight_id = fight_row["fight_id"]
        fight_html = fetch_html(
            fight_row["fight_url"],
            html_dir / f"{fight_id}.html",
            force=force,
            session=session,
            logger=logger,
        )
        if "Checking your browser" in fight_html:
            append_error_row(
                Path("logs") / "errors.csv",
                {
                    "stage": "fight_stats",
                    "entity_id": fight_id,
                    "url": fight_row["fight_url"],
                    "message": "Blocked by browser challenge",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        try:
            rows.extend(parse_fight_stats(fight_html, fight_row))
        except Exception as exc:
            append_error_row(
                Path("logs") / "errors.csv",
                {
                    "stage": "fight_stats_parse",
                    "entity_id": fight_id,
                    "url": fight_row["fight_url"],
                    "message": str(exc),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    scraped_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["scraped_at"] = scraped_at

    new_rows = [row for row in rows if force or (str(row["fight_id"]), str(row["fighter_id"])) not in existing_keys]
    new_frame = pd.DataFrame(new_rows, columns=FIGHT_STATS_COLUMNS)
    if not existing.empty and not force:
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["fight_id", "fighter_id"], keep="last")
    if combined.empty:
        combined = pd.DataFrame(columns=FIGHT_STATS_COLUMNS)

    save_csv(combined, raw_dir / "fight_stats_raw.csv")
    with sqlite3.connect(str(db_path)) as conn:
        combined.to_sql("fight_stats_raw", conn, if_exists="replace", index=False)
    logger.info("Saved %s fight stat rows", len(combined))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape UFCStats per-round fight statistics.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
