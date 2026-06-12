from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.storage import load_csv, save_csv


SNAPSHOT_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "weight_class",
    "scheduled_rounds",
    "age_years",
    "height_cm",
    "reach_cm",
    "stance",
    "ufc_fights",
    "ufc_win_rate",
    "days_since_last_fight",
    "slpm",
    "sapm",
    "strike_differential",
    "td_avg",
    "td_accuracy",
    "td_defense",
    "td_attempt_rate",
    "control_time_per_fight",
    "sub_avg",
    "overall_elo",
    "ufc_minutes",
    "last3_win_rate",
    "last3_slpm",
    "last3_td_avg",
    "finish_rate",
    "sig_strike_accuracy",
    "sig_strike_defense",
    "sig_strike_attempted_per_min",
]


TRAINING_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date",
    "fighter_a_id",
    "fighter_a_name",
    "fighter_b_id",
    "fighter_b_name",
    "weight_class",
    "scheduled_rounds",
    "fighter_a_won",
    "age_diff",
    "height_diff",
    "reach_diff",
    "ufc_fights_diff",
    "ufc_win_rate_diff",
    "days_since_last_fight_diff",
    "slpm_diff",
    "sapm_diff",
    "strike_differential_diff",
    "td_avg_diff",
    "td_accuracy_diff",
    "td_defense_diff",
    "td_attempt_rate_diff",
    "control_time_per_fight_diff",
    "sub_avg_diff",
    "overall_elo_diff",
    "ufc_minutes_diff",
    "last3_win_rate_diff",
    "last3_slpm_diff",
    "last3_td_avg_diff",
    "finish_rate_diff",
    "sig_strike_accuracy_diff",
    "sig_strike_defense_diff",
    "sig_strike_attempted_per_min_diff",
]


def safe_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def safe_diff(left: Any, right: Any) -> float | None:
    left_value = safe_value(left)
    right_value = safe_value(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def sum_stat(rows: list[dict[str, Any]], column: str) -> float:
    total = 0.0
    for row in rows:
        value = safe_value(row.get(column))
        if value is not None:
            total += value
    return total


def build_static_fighter_map(fighters_clean: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if fighters_clean.empty:
        return {}
    fighters = fighters_clean.copy()
    fighters["fighter_id"] = fighters["fighter_id"].astype(str)
    return fighters.set_index("fighter_id").to_dict(orient="index")


def build_stats_lookup(fighter_fight_stats_clean: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if fighter_fight_stats_clean.empty:
        return {}
    stats = fighter_fight_stats_clean.copy()
    stats["fight_id"] = stats["fight_id"].astype(str)
    stats["fighter_id"] = stats["fighter_id"].astype(str)
    return {
        (row["fight_id"], row["fighter_id"]): row.to_dict()
        for _, row in stats.iterrows()
    }


def build_fighter_snapshot(
    fighter_id: str,
    fighter_name: str | None,
    opponent_id: str | None,
    opponent_name: str | None,
    fight_row: pd.Series,
    fighter_static: dict[str, dict[str, Any]],
    official_history: dict[str, list[dict[str, Any]]],
    stat_history: dict[str, list[dict[str, Any]]],
    elo_ratings: dict[str, float],
) -> dict[str, Any]:
    fighter_info = fighter_static.get(fighter_id, {})
    fight_date = pd.to_datetime(fight_row["event_date"], errors="coerce")

    dob = pd.to_datetime(fighter_info.get("dob"), errors="coerce")
    age_years = None
    if pd.notna(fight_date) and pd.notna(dob):
        age_years = (fight_date - dob).days / 365.25

    prior_official = official_history[fighter_id]
    prior_stats = stat_history[fighter_id]
    last_official = prior_official[-1] if prior_official else None
    last3_official = prior_official[-3:]
    last3_stats = prior_stats[-3:]

    official_count = len(prior_official)
    wins = sum(int(row.get("won", 0) or 0) for row in prior_official)
    total_minutes = sum_stat(prior_official, "fight_duration_seconds") / 60.0
    stat_minutes = sum_stat(prior_stats, "fight_duration_seconds") / 60.0
    stat_seconds = sum_stat(prior_stats, "fight_duration_seconds")
    td_attempts = sum_stat(prior_stats, "takedowns_attempted")
    opp_td_attempts = sum_stat(prior_stats, "opponent_takedowns_attempted")
    sig_attempts = sum_stat(prior_stats, "sig_strikes_attempted")
    opp_sig_attempts = sum_stat(prior_stats, "opponent_sig_strikes_attempted")
    finish_wins = sum(
        int((row.get("won", 0) or 0) == 1 and row.get("win_method_group") not in {None, "DEC"})
        for row in prior_official
    )

    days_since_last_fight = None
    if last_official is not None and pd.notna(fight_date):
        prior_date = pd.to_datetime(last_official.get("event_date"), errors="coerce")
        if pd.notna(prior_date):
            days_since_last_fight = int((fight_date - prior_date).days)

    slpm = safe_divide(sum_stat(prior_stats, "sig_strikes_landed"), stat_minutes)
    sapm = safe_divide(sum_stat(prior_stats, "sig_strikes_absorbed"), stat_minutes)
    td_avg = safe_divide(sum_stat(prior_stats, "takedowns_landed") * 15.0, stat_minutes)
    td_accuracy = safe_divide(sum_stat(prior_stats, "takedowns_landed"), td_attempts)
    td_defense = safe_divide(opp_td_attempts - sum_stat(prior_stats, "takedowns_allowed"), opp_td_attempts)
    td_attempt_rate = safe_divide(td_attempts * 15.0, stat_minutes)
    control_time_per_fight = safe_divide(sum_stat(prior_stats, "control_time_seconds"), len(prior_stats))
    sub_avg = safe_divide(sum_stat(prior_stats, "submission_attempts") * 15.0, stat_minutes)
    sig_strike_accuracy = safe_divide(sum_stat(prior_stats, "sig_strikes_landed"), sig_attempts)
    sig_strike_defense = safe_divide(opp_sig_attempts - sum_stat(prior_stats, "sig_strikes_absorbed"), opp_sig_attempts)
    sig_strike_attempted_per_min = safe_divide(sig_attempts, stat_minutes)

    last3_minutes = sum_stat(last3_stats, "fight_duration_seconds") / 60.0
    last3_slpm = safe_divide(sum_stat(last3_stats, "sig_strikes_landed"), last3_minutes)
    last3_td_avg = safe_divide(sum_stat(last3_stats, "takedowns_landed") * 15.0, last3_minutes)

    snapshot = {
        "fight_id": fight_row["fight_id"],
        "event_id": fight_row["event_id"],
        "event_date": fight_row["event_date"],
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "weight_class": fight_row["weight_class"],
        "scheduled_rounds": fight_row["scheduled_rounds"],
        "age_years": age_years,
        "height_cm": safe_value(fighter_info.get("height_cm")),
        "reach_cm": safe_value(fighter_info.get("reach_cm")),
        "stance": fighter_info.get("stance"),
        "ufc_fights": official_count,
        "ufc_win_rate": safe_divide(wins, official_count),
        "days_since_last_fight": days_since_last_fight,
        "slpm": slpm,
        "sapm": sapm,
        "strike_differential": None if slpm is None or sapm is None else slpm - sapm,
        "td_avg": td_avg,
        "td_accuracy": td_accuracy,
        "td_defense": td_defense,
        "td_attempt_rate": td_attempt_rate,
        "control_time_per_fight": control_time_per_fight,
        "sub_avg": sub_avg,
        "overall_elo": elo_ratings[fighter_id],
        "ufc_minutes": total_minutes,
        "last3_win_rate": safe_divide(sum(int(row.get("won", 0) or 0) for row in last3_official), len(last3_official)),
        "last3_slpm": last3_slpm,
        "last3_td_avg": last3_td_avg,
        "finish_rate": safe_divide(finish_wins, wins),
        "sig_strike_accuracy": sig_strike_accuracy,
        "sig_strike_defense": sig_strike_defense,
        "sig_strike_attempted_per_min": sig_strike_attempted_per_min,
    }
    return snapshot


def build_official_history_row(fight_row: pd.Series, fighter_id: str, won: int) -> dict[str, Any]:
    return {
        "fight_id": fight_row["fight_id"],
        "event_date": fight_row["event_date"],
        "fight_duration_seconds": safe_value(fight_row.get("fight_duration_seconds")),
        "won": won,
        "win_method_group": fight_row.get("method_group") if won == 1 else None,
    }


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_ratings: dict[str, float], fighter_a_id: str, fighter_b_id: str, score_a: float, score_b: float, k_factor: float = 32.0) -> None:
    elo_a = elo_ratings[fighter_a_id]
    elo_b = elo_ratings[fighter_b_id]
    expected_a = expected_score(elo_a, elo_b)
    expected_b = expected_score(elo_b, elo_a)
    elo_ratings[fighter_a_id] = elo_a + k_factor * (score_a - expected_a)
    elo_ratings[fighter_b_id] = elo_b + k_factor * (score_b - expected_b)


def build_training_row(fight_row: pd.Series, fighter_a_snapshot: dict[str, Any], fighter_b_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "fight_id": fight_row["fight_id"],
        "event_id": fight_row["event_id"],
        "event_date": fight_row["event_date"],
        "fighter_a_id": fight_row["fighter_a_id"],
        "fighter_a_name": fight_row["fighter_a_name"],
        "fighter_b_id": fight_row["fighter_b_id"],
        "fighter_b_name": fight_row["fighter_b_name"],
        "weight_class": fight_row["weight_class"],
        "scheduled_rounds": fight_row["scheduled_rounds"],
        "fighter_a_won": int(fight_row["fighter_a_won"]),
        "age_diff": safe_diff(fighter_a_snapshot["age_years"], fighter_b_snapshot["age_years"]),
        "height_diff": safe_diff(fighter_a_snapshot["height_cm"], fighter_b_snapshot["height_cm"]),
        "reach_diff": safe_diff(fighter_a_snapshot["reach_cm"], fighter_b_snapshot["reach_cm"]),
        "ufc_fights_diff": safe_diff(fighter_a_snapshot["ufc_fights"], fighter_b_snapshot["ufc_fights"]),
        "ufc_win_rate_diff": safe_diff(fighter_a_snapshot["ufc_win_rate"], fighter_b_snapshot["ufc_win_rate"]),
        "days_since_last_fight_diff": safe_diff(fighter_a_snapshot["days_since_last_fight"], fighter_b_snapshot["days_since_last_fight"]),
        "slpm_diff": safe_diff(fighter_a_snapshot["slpm"], fighter_b_snapshot["slpm"]),
        "sapm_diff": safe_diff(fighter_a_snapshot["sapm"], fighter_b_snapshot["sapm"]),
        "strike_differential_diff": safe_diff(fighter_a_snapshot["strike_differential"], fighter_b_snapshot["strike_differential"]),
        "td_avg_diff": safe_diff(fighter_a_snapshot["td_avg"], fighter_b_snapshot["td_avg"]),
        "td_accuracy_diff": safe_diff(fighter_a_snapshot["td_accuracy"], fighter_b_snapshot["td_accuracy"]),
        "td_defense_diff": safe_diff(fighter_a_snapshot["td_defense"], fighter_b_snapshot["td_defense"]),
        "td_attempt_rate_diff": safe_diff(fighter_a_snapshot["td_attempt_rate"], fighter_b_snapshot["td_attempt_rate"]),
        "control_time_per_fight_diff": safe_diff(fighter_a_snapshot["control_time_per_fight"], fighter_b_snapshot["control_time_per_fight"]),
        "sub_avg_diff": safe_diff(fighter_a_snapshot["sub_avg"], fighter_b_snapshot["sub_avg"]),
        "overall_elo_diff": safe_diff(fighter_a_snapshot["overall_elo"], fighter_b_snapshot["overall_elo"]),
        "ufc_minutes_diff": safe_diff(fighter_a_snapshot["ufc_minutes"], fighter_b_snapshot["ufc_minutes"]),
        "last3_win_rate_diff": safe_diff(fighter_a_snapshot["last3_win_rate"], fighter_b_snapshot["last3_win_rate"]),
        "last3_slpm_diff": safe_diff(fighter_a_snapshot["last3_slpm"], fighter_b_snapshot["last3_slpm"]),
        "last3_td_avg_diff": safe_diff(fighter_a_snapshot["last3_td_avg"], fighter_b_snapshot["last3_td_avg"]),
        "finish_rate_diff": safe_diff(fighter_a_snapshot["finish_rate"], fighter_b_snapshot["finish_rate"]),
        "sig_strike_accuracy_diff": safe_diff(fighter_a_snapshot["sig_strike_accuracy"], fighter_b_snapshot["sig_strike_accuracy"]),
        "sig_strike_defense_diff": safe_diff(fighter_a_snapshot["sig_strike_defense"], fighter_b_snapshot["sig_strike_defense"]),
        "sig_strike_attempted_per_min_diff": safe_diff(
            fighter_a_snapshot["sig_strike_attempted_per_min"],
            fighter_b_snapshot["sig_strike_attempted_per_min"],
        ),
    }


def build_training_schema(
    fights_clean: pd.DataFrame,
    fighters_clean: pd.DataFrame,
    fighter_fight_stats_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fights_clean.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS), pd.DataFrame(columns=TRAINING_COLUMNS)

    fighter_static = build_static_fighter_map(fighters_clean)
    stats_lookup = build_stats_lookup(fighter_fight_stats_clean)
    official_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stat_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    elo_ratings: dict[str, float] = defaultdict(lambda: 1500.0)

    fights = fights_clean.copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    fights = fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    snapshots: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for _, fight_row in fights.iterrows():
        fighter_a_id = str(fight_row["fighter_a_id"])
        fighter_b_id = str(fight_row["fighter_b_id"])

        fighter_a_snapshot = build_fighter_snapshot(
            fighter_id=fighter_a_id,
            fighter_name=fight_row.get("fighter_a_name"),
            opponent_id=fighter_b_id,
            opponent_name=fight_row.get("fighter_b_name"),
            fight_row=fight_row,
            fighter_static=fighter_static,
            official_history=official_history,
            stat_history=stat_history,
            elo_ratings=elo_ratings,
        )
        fighter_b_snapshot = build_fighter_snapshot(
            fighter_id=fighter_b_id,
            fighter_name=fight_row.get("fighter_b_name"),
            opponent_id=fighter_a_id,
            opponent_name=fight_row.get("fighter_a_name"),
            fight_row=fight_row,
            fighter_static=fighter_static,
            official_history=official_history,
            stat_history=stat_history,
            elo_ratings=elo_ratings,
        )
        snapshots.extend([fighter_a_snapshot, fighter_b_snapshot])

        is_decisive = (
            pd.notna(fight_row.get("fighter_a_won"))
            and fight_row.get("method_group") not in {"DRAW", "NC"}
            and pd.notna(fight_row.get("event_date"))
        )
        if is_decisive:
            training_rows.append(build_training_row(fight_row, fighter_a_snapshot, fighter_b_snapshot))

            fighter_a_won = int(fight_row["fighter_a_won"])
            fighter_b_won = 1 - fighter_a_won

            official_history[fighter_a_id].append(build_official_history_row(fight_row, fighter_a_id, fighter_a_won))
            official_history[fighter_b_id].append(build_official_history_row(fight_row, fighter_b_id, fighter_b_won))

            fighter_a_stats = stats_lookup.get((str(fight_row["fight_id"]), fighter_a_id))
            fighter_b_stats = stats_lookup.get((str(fight_row["fight_id"]), fighter_b_id))
            if fighter_a_stats is not None:
                stat_history[fighter_a_id].append(fighter_a_stats)
            if fighter_b_stats is not None:
                stat_history[fighter_b_id].append(fighter_b_stats)

            update_elo(
                elo_ratings=elo_ratings,
                fighter_a_id=fighter_a_id,
                fighter_b_id=fighter_b_id,
                score_a=float(fighter_a_won),
                score_b=float(fighter_b_won),
            )

    snapshots_df = pd.DataFrame(snapshots, columns=SNAPSHOT_COLUMNS)
    training_df = pd.DataFrame(training_rows, columns=TRAINING_COLUMNS)
    if not snapshots_df.empty:
        snapshots_df["event_date"] = snapshots_df["event_date"].dt.date.astype(str)
    if not training_df.empty:
        training_df["event_date"] = training_df["event_date"].dt.date.astype(str)
    return snapshots_df, training_df


def main() -> None:
    clean_dir = Path("data") / "current_model" / "clean"
    features_dir = Path("data") / "current_model" / "features"
    db_path = Path("ufc_stats.db")

    fights_clean = load_csv(clean_dir / "fights_clean.csv")
    fighters_clean = load_csv(clean_dir / "fighters_clean.csv")
    fighter_fight_stats_clean = load_csv(clean_dir / "fighter_fight_stats_clean.csv")

    snapshots_df, training_df = build_training_schema(
        fights_clean=fights_clean,
        fighters_clean=fighters_clean,
        fighter_fight_stats_clean=fighter_fight_stats_clean,
    )

    save_csv(snapshots_df, features_dir / "fighter_snapshots.csv")
    save_csv(training_df, features_dir / "ml_training_schema.csv")

    with sqlite3.connect(str(db_path)) as conn:
        snapshots_df.to_sql("fighter_snapshots", conn, if_exists="replace", index=False)
        training_df.to_sql("ml_training_schema", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
