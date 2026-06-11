from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.utils.storage import load_csv, save_csv


def add_result(
    results: list[dict[str, object]],
    check: str,
    passed: bool,
    level: str,
    count: int | None = None,
    details: str | None = None,
) -> None:
    results.append({"check": check, "passed": passed, "level": level, "count": count, "details": details})


def main() -> int:
    clean_dir = Path("data") / "current_model" / "clean"
    report_path = Path("validation_report.csv")
    summary_path = Path("validation_summary.txt")

    events = load_csv(clean_dir / "events_clean.csv")
    fighters = load_csv(clean_dir / "fighters_clean.csv")
    fights = load_csv(clean_dir / "fights_clean.csv")
    fighter_stats = load_csv(clean_dir / "fighter_fight_stats_clean.csv")
    fighters_raw = load_csv(Path("data") / "current_model" / "raw" / "fighters_raw.csv")

    results: list[dict[str, object]] = []
    critical_failures = 0
    warnings = 0

    add_result(results, "total_events", True, "info", int(len(events)))
    add_result(results, "total_fights", True, "info", int(len(fights)))
    add_result(results, "total_fighters", True, "info", int(len(fighters)))

    missing_dob = int(fighters["dob"].isna().sum()) if "dob" in fighters else 0
    missing_reach = int(fighters["reach_cm"].isna().sum()) if "reach_cm" in fighters else 0
    add_result(results, "missing_dob", missing_dob == 0, "warning" if missing_dob else "info", missing_dob)
    add_result(results, "missing_reach", missing_reach == 0, "warning" if missing_reach else "info", missing_reach)
    warnings += int(missing_dob > 0) + int(missing_reach > 0)

    if not fights.empty:
        duplicate_fights = int(fights.duplicated(subset=["fight_id"]).sum()) if "fight_id" in fights else 0
        add_result(results, "duplicate_fight_ids", duplicate_fights == 0, "critical" if duplicate_fights else "info", duplicate_fights)
        critical_failures += int(duplicate_fights > 0)

        if not fighter_stats.empty and "fight_id" in fighter_stats:
            grouped_fights = fighter_stats.groupby("fight_id").size()
            wrong_fight_counts = int((grouped_fights != 2).sum())
        else:
            wrong_fight_counts = 0
        add_result(results, "exactly_two_fighters_per_stat_fight", wrong_fight_counts == 0, "critical" if wrong_fight_counts else "info", wrong_fight_counts)
        critical_failures += int(wrong_fight_counts > 0)

        completed_mask = fights.get("is_completed", pd.Series(0, index=fights.index)).fillna(0).astype(int) == 1
        result_mask = fights.get("has_result", pd.Series(0, index=fights.index)).fillna(0).astype(int) == 1
        decisive_mask = ~fights.get("method_group", pd.Series(dtype=object)).isin(["DRAW", "NC"])
        winners_missing = fights[completed_mask & result_mask & decisive_mask & fights["winner_id"].isna()]
        add_result(results, "winner_present_for_completed_decisive_fights", len(winners_missing) == 0, "critical" if len(winners_missing) else "info", int(len(winners_missing)))
        critical_failures += int(len(winners_missing) > 0)

    if not fighter_stats.empty:
        duplicate_rows = int(fighter_stats.duplicated(subset=["fight_id", "fighter_id"]).sum()) if {"fight_id", "fighter_id"}.issubset(fighter_stats.columns) else 0
        add_result(results, "duplicate_fighter_fight_rows", duplicate_rows == 0, "critical" if duplicate_rows else "info", duplicate_rows)
        critical_failures += int(duplicate_rows > 0)

        if "fight_order" in fighter_stats:
            order_issues = 0
            for _, group in fighter_stats.groupby("fighter_id"):
                ordered = sorted(group["fight_order"].dropna().astype(int).tolist())
                if ordered != list(range(1, len(ordered) + 1)):
                    order_issues += 1
            add_result(results, "fight_order_sequential", order_issues == 0, "critical" if order_issues else "info", order_issues)
            critical_failures += int(order_issues > 0)

        stat_checks = {
            "sig_strikes_landed_le_attempted": (fighter_stats["sig_strikes_landed"] <= fighter_stats["sig_strikes_attempted"]).all() if {"sig_strikes_landed", "sig_strikes_attempted"}.issubset(fighter_stats.columns) else True,
            "total_strikes_landed_le_attempted": (fighter_stats["total_strikes_landed"] <= fighter_stats["total_strikes_attempted"]).all() if {"total_strikes_landed", "total_strikes_attempted"}.issubset(fighter_stats.columns) else True,
            "takedowns_landed_le_attempted": (fighter_stats["takedowns_landed"] <= fighter_stats["takedowns_attempted"]).all() if {"takedowns_landed", "takedowns_attempted"}.issubset(fighter_stats.columns) else True,
            "control_time_non_negative": (fighter_stats["control_time_seconds"].fillna(0) >= 0).all() if "control_time_seconds" in fighter_stats else True,
            "fight_duration_positive": (fighter_stats["fight_duration_seconds"].fillna(0) > 0).all() if "fight_duration_seconds" in fighter_stats else True,
        }
        for name, passed in stat_checks.items():
            add_result(results, name, bool(passed), "critical" if not passed else "info", None)
            critical_failures += int(not passed)

    if not fighters_raw.empty:
        for column in ["profile_str_acc", "profile_str_def", "profile_TD_acc", "profile_TD_def"]:
            if column in fighters_raw.columns:
                invalid = fighters_raw[column].dropna().map(lambda value: not (0 <= float(value) <= 1)).sum()
                add_result(results, f"{column}_range", invalid == 0, "critical" if invalid else "info", int(invalid))
                critical_failures += int(invalid > 0)

    report = pd.DataFrame(results)
    save_csv(report, report_path)

    summary_lines = [
        f"Total events: {len(events)}",
        f"Total fights: {len(fights)}",
        f"Total fighters: {len(fighters)}",
        f"Missing DOB count: {missing_dob}",
        f"Missing reach count: {missing_reach}",
        f"Failed validations: {critical_failures}",
        f"Warnings: {warnings}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
