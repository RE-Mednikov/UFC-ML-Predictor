from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.clean.parsers import (
    parse_control_time_to_seconds,
    parse_date_to_iso,
    parse_fight_duration,
    parse_float,
    parse_height_to_cm,
    parse_int,
    parse_name_parts,
    parse_method_group,
    parse_percentage,
    parse_reach_to_cm,
    parse_record,
    parse_time_to_seconds,
    parse_weight_to_lbs,
    split_location,
)
from src.current_model.db.schema import initialize_database
from src.current_model.utils.storage import load_csv, save_csv


def build_fighter_name_map(fights_raw: pd.DataFrame) -> dict[str, str]:
    if fights_raw.empty:
        return {}

    name_map: dict[str, str] = {}
    for fighter_id_column, fighter_name_column in (
        ("fighter_1_id", "fighter_1_name"),
        ("fighter_2_id", "fighter_2_name"),
    ):
        pairs = fights_raw[[fighter_id_column, fighter_name_column]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            fighter_id = str(row[fighter_id_column]).strip()
            fighter_name = str(row[fighter_name_column]).strip()
            if fighter_id and fighter_name:
                name_map[fighter_id] = fighter_name
    return name_map


def clean_fighter_name(full_name: object, fallback_name: str | None = None) -> str | None:
    if fallback_name:
        return fallback_name

    if full_name is None or pd.isna(full_name):
        return None

    text = str(full_name).strip()
    if not text:
        return None
    if " Record:" in text:
        text = text.split(" Record:", 1)[0].strip()
    return text or None


def build_events_clean(events_raw: pd.DataFrame) -> pd.DataFrame:
    if events_raw.empty:
        return pd.DataFrame(
            columns=["event_id", "event_url", "event_name", "event_date", "city", "region", "country", "location_raw"]
        )

    rows = []
    for _, row in events_raw.iterrows():
        city, region, country = split_location(row.get("event_location_raw"))
        rows.append(
            {
                "event_id": row.get("event_id"),
                "event_url": row.get("event_url"),
                "event_name": row.get("event_name"),
                "event_date": parse_date_to_iso(row.get("event_date_raw")),
                "city": city,
                "region": region,
                "country": country,
                "location_raw": row.get("event_location_raw"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["event_id"])


def build_fighters_clean(fighters_raw: pd.DataFrame, fights_raw: pd.DataFrame) -> pd.DataFrame:
    if fighters_raw.empty:
        return pd.DataFrame(
            columns=[
                "fighter_id",
                "fighter_url",
                "fighter_name",
                "first_name",
                "last_name",
                "nickname",
                "height_cm",
                "weight_lbs",
                "reach_cm",
                "stance",
                "dob",
                "profile_record_wins",
                "profile_record_losses",
                "profile_record_draws",
                "profile_slpm",
                "profile_str_acc",
                "profile_sapm",
                "profile_str_def",
                "profile_td_avg",
                "profile_td_acc",
                "profile_td_def",
                "profile_sub_avg",
            ]
        )

    fighter_name_map = build_fighter_name_map(fights_raw)
    rows = []
    for _, row in fighters_raw.iterrows():
        wins, losses, draws = parse_record(row.get("record_raw"))
        fighter_id = row.get("fighter_id")
        canonical_name = clean_fighter_name(row.get("full_name"), fighter_name_map.get(str(fighter_id)))
        first_name, last_name = parse_name_parts(canonical_name)
        rows.append(
            {
                "fighter_id": fighter_id,
                "fighter_url": row.get("fighter_url"),
                "fighter_name": canonical_name,
                "first_name": first_name,
                "last_name": last_name,
                "nickname": row.get("nickname"),
                "height_cm": parse_height_to_cm(row.get("height_raw")),
                "weight_lbs": parse_weight_to_lbs(row.get("weight_raw")),
                "reach_cm": parse_reach_to_cm(row.get("reach_raw")),
                "stance": row.get("stance"),
                "dob": parse_date_to_iso(row.get("dob_raw")),
                "profile_record_wins": wins,
                "profile_record_losses": losses,
                "profile_record_draws": draws,
                "profile_slpm": parse_float(row.get("profile_SLpM")),
                "profile_str_acc": parse_percentage(row.get("profile_str_acc")),
                "profile_sapm": parse_float(row.get("profile_SApM")),
                "profile_str_def": parse_percentage(row.get("profile_str_def")),
                "profile_td_avg": parse_float(row.get("profile_TD_avg")),
                "profile_td_acc": parse_percentage(row.get("profile_TD_acc")),
                "profile_td_def": parse_percentage(row.get("profile_TD_def")),
                "profile_sub_avg": parse_float(row.get("profile_sub_avg")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["fighter_id"])


def build_fights_clean(fights_raw: pd.DataFrame, events_clean: pd.DataFrame) -> pd.DataFrame:
    if fights_raw.empty:
        return pd.DataFrame(
            columns=[
                "fight_id",
                "fight_url",
                "event_id",
                "event_date",
                "event_name",
                "fighter_a_id",
                "fighter_a_name",
                "fighter_b_id",
                "fighter_b_name",
                "winner_id",
                "loser_id",
                "fighter_a_won",
                "weight_class",
                "scheduled_rounds",
                "method_group",
                "method_raw",
                "method_details",
                "ending_round",
                "ending_time_seconds",
                "fight_duration_seconds",
                "referee",
                "has_result",
                "is_completed",
            ]
        )

    event_dates = events_clean.set_index("event_id")["event_date"].to_dict() if not events_clean.empty else {}
    rows = []
    for _, row in fights_raw.iterrows():
        fighter_a_id = row.get("fighter_1_id")
        fighter_b_id = row.get("fighter_2_id")
        winner_id = row.get("winner_id")
        method_raw = row.get("method_raw")
        method_details = row.get("method_details")
        method_group = parse_method_group(method_raw)
        method_raw_text = str(method_raw or "").upper()
        method_details_text = str(method_details or "").upper()
        if pd.isna(winner_id) or winner_id in {"", "nan", "None"}:
            if method_group == "DEC" or "TIME EXPIRED" in method_details_text:
                method_group = "DRAW"
            elif "OVERTURN" in method_raw_text or "COULD NOT CONTINUE" in method_raw_text:
                method_group = "NC"
        ending_round = parse_int(row.get("ending_round"))
        ending_time_seconds = parse_time_to_seconds(row.get("ending_time_raw"))
        scheduled_rounds = parse_int(row.get("scheduled_rounds"))
        fight_duration_seconds = parse_fight_duration(
            ending_round,
            ending_time_seconds,
            scheduled_rounds,
            method_group,
        )
        has_result = bool(winner_id) or method_group in {"DRAW", "NC"}
        loser_id = fighter_b_id if winner_id == fighter_a_id else fighter_a_id if winner_id == fighter_b_id else None
        rows.append(
            {
                "fight_id": row.get("fight_id"),
                "fight_url": row.get("fight_url"),
                "event_id": row.get("event_id"),
                "event_date": event_dates.get(row.get("event_id")) or parse_date_to_iso(row.get("event_date_raw")),
                "event_name": row.get("event_name"),
                "fighter_a_id": fighter_a_id,
                "fighter_a_name": row.get("fighter_1_name"),
                "fighter_b_id": fighter_b_id,
                "fighter_b_name": row.get("fighter_2_name"),
                "winner_id": None if method_group in {"DRAW", "NC"} else winner_id,
                "loser_id": loser_id,
                "fighter_a_won": 1 if winner_id == fighter_a_id else 0 if winner_id == fighter_b_id else None,
                "weight_class": row.get("weight_class"),
                "scheduled_rounds": scheduled_rounds,
                "method_group": method_group,
                "method_raw": method_raw,
                "method_details": method_details,
                "ending_round": ending_round,
                "ending_time_seconds": ending_time_seconds,
                "fight_duration_seconds": fight_duration_seconds,
                "referee": row.get("referee"),
                "has_result": int(has_result),
                "is_completed": int(has_result and fight_duration_seconds is not None),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["fight_id"])


def build_fighter_fight_stats_clean(fight_stats_raw: pd.DataFrame, fights_clean: pd.DataFrame) -> pd.DataFrame:
    if fight_stats_raw.empty or fights_clean.empty:
        return pd.DataFrame(
            columns=[
                "fight_id",
                "event_id",
                "event_date",
                "fighter_id",
                "opponent_id",
                "fight_order",
                "won",
                "lost",
                "draw",
                "no_contest",
                "fight_duration_seconds",
                "scheduled_rounds",
                "knockdowns",
                "sig_strikes_landed",
                "sig_strikes_attempted",
                "sig_strikes_absorbed",
                "opponent_sig_strikes_attempted",
                "total_strikes_landed",
                "total_strikes_attempted",
                "total_strikes_absorbed",
                "takedowns_landed",
                "takedowns_attempted",
                "takedowns_allowed",
                "opponent_takedowns_attempted",
                "submission_attempts",
                "reversals",
                "control_time_seconds",
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
                "win_method_group",
                "loss_method_group",
            ]
        )

    numeric_columns = [
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
    available_numeric_columns = [column for column in numeric_columns if column in fight_stats_raw.columns]

    stats = fight_stats_raw.copy()
    if "control_time_raw" in stats.columns:
        stats["control_time_seconds"] = stats["control_time_raw"].map(parse_control_time_to_seconds)
    elif "control_time_seconds" not in stats.columns:
        stats["control_time_seconds"] = pd.NA

    aggregation_map = {column: "sum" for column in available_numeric_columns}
    aggregation_map["control_time_seconds"] = "sum"
    grouped = stats.groupby(["fight_id", "fighter_id"], dropna=False).agg(aggregation_map).reset_index()

    long_rows = []
    fight_columns = [
        "fight_id",
        "event_id",
        "event_date",
        "fighter_a_id",
        "fighter_b_id",
        "winner_id",
        "method_group",
        "fight_duration_seconds",
        "scheduled_rounds",
    ]
    for _, row in fights_clean[fight_columns].iterrows():
        base = row.to_dict()
        long_rows.append({**base, "fighter_id": row["fighter_a_id"], "opponent_id": row["fighter_b_id"]})
        long_rows.append({**base, "fighter_id": row["fighter_b_id"], "opponent_id": row["fighter_a_id"]})
    long_pairs = pd.DataFrame(long_rows)

    fights_with_stats = set(grouped["fight_id"].astype(str))
    long_pairs = long_pairs[long_pairs["fight_id"].astype(str).isin(fights_with_stats)].copy()

    clean = long_pairs.merge(grouped, on=["fight_id", "fighter_id"], how="left")
    opponent_stats = grouped.rename(
        columns={
            "fighter_id": "opponent_id",
            "knockdowns": "opponent_knockdowns",
            "sig_strikes_landed": "sig_strikes_absorbed",
            "sig_strikes_attempted": "opponent_sig_strikes_attempted",
            "total_strikes_landed": "total_strikes_absorbed",
            "total_strikes_attempted": "opponent_total_strikes_attempted",
            "takedowns_landed": "takedowns_allowed",
            "takedowns_attempted": "opponent_takedowns_attempted",
        }
    )
    opponent_columns = [
        "fight_id",
        "opponent_id",
        "sig_strikes_absorbed",
        "opponent_sig_strikes_attempted",
        "total_strikes_absorbed",
        "takedowns_allowed",
        "opponent_takedowns_attempted",
    ]
    clean = clean.merge(opponent_stats[opponent_columns], on=["fight_id", "opponent_id"], how="left")

    clean["won"] = (clean["fighter_id"] == clean["winner_id"]).astype(int)
    clean["lost"] = ((clean["winner_id"].notna()) & (clean["fighter_id"] != clean["winner_id"]) & (clean["method_group"] != "DRAW") & (clean["method_group"] != "NC")).astype(int)
    clean["draw"] = (clean["method_group"] == "DRAW").astype(int)
    clean["no_contest"] = (clean["method_group"] == "NC").astype(int)
    clean["win_method_group"] = clean["method_group"].where(clean["won"] == 1)
    clean["loss_method_group"] = clean["method_group"].where(clean["lost"] == 1)

    clean = clean.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    clean["fight_order"] = clean.groupby("fighter_id").cumcount() + 1

    desired_columns = [
        "fight_id",
        "event_id",
        "event_date",
        "fighter_id",
        "opponent_id",
        "fight_order",
        "won",
        "lost",
        "draw",
        "no_contest",
        "fight_duration_seconds",
        "scheduled_rounds",
        "knockdowns",
        "sig_strikes_landed",
        "sig_strikes_attempted",
        "sig_strikes_absorbed",
        "opponent_sig_strikes_attempted",
        "total_strikes_landed",
        "total_strikes_attempted",
        "total_strikes_absorbed",
        "takedowns_landed",
        "takedowns_attempted",
        "takedowns_allowed",
        "opponent_takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_time_seconds",
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
        "win_method_group",
        "loss_method_group",
    ]
    existing_columns = [column for column in desired_columns if column in clean.columns]
    return clean[existing_columns].drop_duplicates(subset=["fight_id", "fighter_id"])


def main() -> None:
    data_dir = Path("data") / "current_model"
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    db_path = Path("ufc_stats.db")

    clean_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(str(db_path))

    events_raw = load_csv(raw_dir / "events_raw.csv")
    fights_raw = load_csv(raw_dir / "fights_raw.csv")
    fight_stats_raw = load_csv(raw_dir / "fight_stats_raw.csv")
    fighters_raw = load_csv(raw_dir / "fighters_raw.csv")

    events_clean = build_events_clean(events_raw)
    fighters_clean = build_fighters_clean(fighters_raw, fights_raw)
    fights_clean = build_fights_clean(fights_raw, events_clean)
    fighter_fight_stats_clean = build_fighter_fight_stats_clean(fight_stats_raw, fights_clean)

    save_csv(events_clean, clean_dir / "events_clean.csv")
    save_csv(fighters_clean, clean_dir / "fighters_clean.csv")
    save_csv(fights_clean, clean_dir / "fights_clean.csv")
    save_csv(fighter_fight_stats_clean, clean_dir / "fighter_fight_stats_clean.csv")

    with sqlite3.connect(str(db_path)) as conn:
        events_clean.to_sql("events_clean", conn, if_exists="replace", index=False)
        fighters_clean.to_sql("fighters_clean", conn, if_exists="replace", index=False)
        fights_clean.to_sql("fights_clean", conn, if_exists="replace", index=False)
        fighter_fight_stats_clean.to_sql("fighter_fight_stats_clean", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
