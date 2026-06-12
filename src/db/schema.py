from __future__ import annotations

import sqlite3


RAW_TABLE_SCHEMAS = {
    "events_raw": """
        CREATE TABLE IF NOT EXISTS events_raw (
            event_id TEXT PRIMARY KEY,
            event_url TEXT,
            event_name TEXT,
            event_date_raw TEXT,
            event_location_raw TEXT,
            scraped_at TEXT
        )
    """,
    "fights_raw": """
        CREATE TABLE IF NOT EXISTS fights_raw (
            fight_id TEXT PRIMARY KEY,
            fight_url TEXT,
            event_id TEXT,
            event_name TEXT,
            event_date_raw TEXT,
            fighter_1_name TEXT,
            fighter_1_url TEXT,
            fighter_1_id TEXT,
            fighter_2_name TEXT,
            fighter_2_url TEXT,
            fighter_2_id TEXT,
            winner_name TEXT,
            winner_id TEXT,
            weight_class TEXT,
            method_raw TEXT,
            method_details TEXT,
            ending_round INTEGER,
            ending_time_raw TEXT,
            scheduled_rounds INTEGER,
            referee TEXT,
            scraped_at TEXT
        )
    """,
    "fight_stats_raw": """
        CREATE TABLE IF NOT EXISTS fight_stats_raw (
            fight_id TEXT,
            event_id TEXT,
            event_date_raw TEXT,
            round INTEGER,
            fighter_id TEXT,
            fighter_name TEXT,
            opponent_id TEXT,
            opponent_name TEXT,
            knockdowns INTEGER,
            sig_strikes_landed INTEGER,
            sig_strikes_attempted INTEGER,
            total_strikes_landed INTEGER,
            total_strikes_attempted INTEGER,
            takedowns_landed INTEGER,
            takedowns_attempted INTEGER,
            submission_attempts INTEGER,
            reversals INTEGER,
            control_time_raw TEXT,
            head_landed INTEGER,
            head_attempted INTEGER,
            body_landed INTEGER,
            body_attempted INTEGER,
            leg_landed INTEGER,
            leg_attempted INTEGER,
            distance_landed INTEGER,
            distance_attempted INTEGER,
            clinch_landed INTEGER,
            clinch_attempted INTEGER,
            ground_landed INTEGER,
            ground_attempted INTEGER,
            scraped_at TEXT
        )
    """,
    "fighters_raw": """
        CREATE TABLE IF NOT EXISTS fighters_raw (
            fighter_id TEXT PRIMARY KEY,
            fighter_url TEXT,
            first_name TEXT,
            last_name TEXT,
            full_name TEXT,
            nickname TEXT,
            height_raw TEXT,
            weight_raw TEXT,
            reach_raw TEXT,
            stance TEXT,
            dob_raw TEXT,
            record_raw TEXT,
            profile_SLpM REAL,
            profile_str_acc REAL,
            profile_SApM REAL,
            profile_str_def REAL,
            profile_TD_avg REAL,
            profile_TD_acc REAL,
            profile_TD_def REAL,
            profile_sub_avg REAL,
            scraped_at TEXT
        )
    """,
}


CLEAN_TABLE_SCHEMAS = {
    "events_clean": """
        CREATE TABLE IF NOT EXISTS events_clean (
            event_id TEXT PRIMARY KEY,
            event_url TEXT,
            event_name TEXT,
            event_date TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            location_raw TEXT
        )
    """,
    "fighters_clean": """
        CREATE TABLE IF NOT EXISTS fighters_clean (
            fighter_id TEXT PRIMARY KEY,
            fighter_url TEXT,
            fighter_name TEXT,
            height_cm REAL,
            weight_lbs INTEGER,
            reach_cm REAL,
            stance TEXT,
            dob TEXT,
            profile_record_wins INTEGER,
            profile_record_losses INTEGER,
            profile_record_draws INTEGER
        )
    """,
    "fights_clean": """
        CREATE TABLE IF NOT EXISTS fights_clean (
            fight_id TEXT PRIMARY KEY,
            fight_url TEXT,
            event_id TEXT,
            event_date TEXT,
            fighter_a_id TEXT,
            fighter_a_name TEXT,
            fighter_b_id TEXT,
            fighter_b_name TEXT,
            winner_id TEXT,
            loser_id TEXT,
            fighter_a_won INTEGER,
            weight_class TEXT,
            scheduled_rounds INTEGER,
            method_group TEXT,
            method_raw TEXT,
            method_details TEXT,
            ending_round INTEGER,
            ending_time_seconds INTEGER,
            fight_duration_seconds INTEGER,
            referee TEXT
        )
    """,
    "fighter_fight_stats_clean": """
        CREATE TABLE IF NOT EXISTS fighter_fight_stats_clean (
            fight_id TEXT,
            event_id TEXT,
            event_date TEXT,
            fighter_id TEXT,
            opponent_id TEXT,
            fight_order INTEGER,
            won INTEGER,
            lost INTEGER,
            draw INTEGER,
            no_contest INTEGER,
            fight_duration_seconds INTEGER,
            scheduled_rounds INTEGER,
            knockdowns INTEGER,
            sig_strikes_landed INTEGER,
            sig_strikes_attempted INTEGER,
            sig_strikes_absorbed INTEGER,
            opponent_sig_strikes_attempted INTEGER,
            total_strikes_landed INTEGER,
            total_strikes_attempted INTEGER,
            total_strikes_absorbed INTEGER,
            takedowns_landed INTEGER,
            takedowns_attempted INTEGER,
            takedowns_allowed INTEGER,
            opponent_takedowns_attempted INTEGER,
            submission_attempts INTEGER,
            reversals INTEGER,
            control_time_seconds INTEGER,
            head_landed INTEGER,
            head_attempted INTEGER,
            body_landed INTEGER,
            body_attempted INTEGER,
            leg_landed INTEGER,
            leg_attempted INTEGER,
            distance_landed INTEGER,
            distance_attempted INTEGER,
            clinch_landed INTEGER,
            clinch_attempted INTEGER,
            ground_landed INTEGER,
            ground_attempted INTEGER,
            win_method_group TEXT,
            loss_method_group TEXT,
            PRIMARY KEY (fight_id, fighter_id)
        )
    """,
}


def initialize_database(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for statement in RAW_TABLE_SCHEMAS.values():
            conn.execute(statement)
        for statement in CLEAN_TABLE_SCHEMAS.values():
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()