from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.storage import load_csv, save_csv


ID_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date",
    "fighter_a_id",
    "fighter_a_name",
    "fighter_b_id",
    "fighter_b_name",
]

BASE_FEATURE_COLUMNS = [
    "scheduled_rounds",
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

META_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date",
    "event_name",
    "fighter_a_id",
    "fighter_a_name",
    "fighter_b_id",
    "fighter_b_name",
    "split",
]

TARGET_COLUMN = "fighter_a_won"
MIN_TRAINING_DATE = pd.Timestamp("2010-01-01")
NON_DIRECTIONAL_FEATURE_COLUMNS = [
    "scheduled_rounds",
    "is_title_fight",
    "is_women_fight",
]


def slugify_weight_class(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_weight_class(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"

    if "women" in text and "strawweight" in text:
        return "women_s_strawweight"
    if "women" in text and "flyweight" in text:
        return "women_s_flyweight"
    if "women" in text and "bantamweight" in text:
        return "women_s_bantamweight"
    if "women" in text and "featherweight" in text:
        return "women_s_featherweight"
    if "flyweight" in text:
        return "flyweight"
    if "bantamweight" in text:
        return "bantamweight"
    if "featherweight" in text:
        return "featherweight"
    if "lightweight" in text:
        return "lightweight"
    if "welterweight" in text:
        return "welterweight"
    if "middleweight" in text:
        return "middleweight"
    if "light heavyweight" in text:
        return "light_heavyweight"
    if "heavyweight" in text and "super" not in text:
        return "heavyweight"
    if "super heavyweight" in text:
        return "super_heavyweight"
    if "open weight" in text:
        return "open_weight"
    if "catch weight" in text:
        return "catch_weight"
    return "unknown"


def assign_chronological_splits(df: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.15) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)

    dated = df.sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)
    event_frame = (
        dated.groupby(["event_id", "event_date"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values(["event_date", "event_id"])
        .reset_index(drop=True)
    )

    total_rows = int(event_frame["row_count"].sum())
    train_target = total_rows * train_ratio
    validation_target = total_rows * (train_ratio + validation_ratio)

    cumulative_rows = 0
    splits: list[str] = []
    for _, row in event_frame.iterrows():
        if cumulative_rows < train_target:
            split = "train"
        elif cumulative_rows < validation_target:
            split = "validation"
        else:
            split = "test"
        splits.append(split)
        cumulative_rows += int(row["row_count"])

    event_frame["split"] = splits
    dated = dated.merge(event_frame[["event_id", "split"]], on="event_id", how="left")
    return dated["split"].reset_index(drop=True)


def build_symmetric_training_rows(df: pd.DataFrame, directional_feature_columns: list[str], weight_dummy_columns: list[str]) -> pd.DataFrame:
    forward = df.copy()

    reverse = df.copy()
    reverse["fighter_a_id"] = df["fighter_b_id"]
    reverse["fighter_a_name"] = df["fighter_b_name"]
    reverse["fighter_b_id"] = df["fighter_a_id"]
    reverse["fighter_b_name"] = df["fighter_a_name"]
    reverse[TARGET_COLUMN] = 1 - reverse[TARGET_COLUMN]
    for column in directional_feature_columns:
        reverse[column] = -reverse[column]

    output_columns = (
        META_COLUMNS
        + [TARGET_COLUMN]
        + NON_DIRECTIONAL_FEATURE_COLUMNS
        + directional_feature_columns
        + weight_dummy_columns
    )
    symmetric = pd.concat([forward[output_columns], reverse[output_columns]], ignore_index=True)
    return symmetric


def select_output_columns(df: pd.DataFrame, directional_feature_columns: list[str], weight_dummy_columns: list[str]) -> list[str]:
    return (
        META_COLUMNS
        + [TARGET_COLUMN]
        + NON_DIRECTIONAL_FEATURE_COLUMNS
        + directional_feature_columns
        + weight_dummy_columns
    )


def build_training_data(training_schema: pd.DataFrame) -> pd.DataFrame:
    if training_schema.empty:
        return pd.DataFrame()

    df = training_schema.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df[df["fighter_a_won"].isin([0, 1])].copy()
    df = df[df["event_date"] >= MIN_TRAINING_DATE].copy()
    df = df.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    df["split"] = assign_chronological_splits(df)

    df["weight_class"] = df["weight_class"].fillna("Unknown")
    df["weight_class_primary"] = df["weight_class"].map(normalize_weight_class)
    df["is_title_fight"] = df["weight_class"].str.contains("title", case=False, na=False).astype(int)
    df["is_women_fight"] = df["weight_class"].str.contains("women", case=False, na=False).astype(int)

    weight_dummies = pd.get_dummies(df["weight_class_primary"].map(slugify_weight_class), prefix="weight_class", dtype=int)
    df = pd.concat([df, weight_dummies], axis=1)
    directional_feature_columns = BASE_FEATURE_COLUMNS[1:]
    weight_dummy_columns = sorted(weight_dummies.columns.tolist())

    output_columns = select_output_columns(df, directional_feature_columns, weight_dummy_columns)

    split_frames: list[pd.DataFrame] = []
    for split_name, split_df in df.groupby("split", sort=False):
        if split_name == "train":
            split_output = build_symmetric_training_rows(
                df=split_df,
                directional_feature_columns=directional_feature_columns,
                weight_dummy_columns=weight_dummy_columns,
            )
        else:
            split_output = split_df[output_columns].copy()
        split_frames.append(split_output)

    return pd.concat(split_frames, ignore_index=True)


def build_split_frames(training_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if training_data.empty:
        return {"train": training_data.copy(), "validation": training_data.copy(), "test": training_data.copy()}
    split_frames: dict[str, pd.DataFrame] = {}
    for split, frame in training_data.groupby("split", dropna=False):
        split_frames[split] = frame.drop(columns=["split"], errors="ignore").reset_index(drop=True)
    return split_frames


def main() -> None:
    features_dir = Path("data") / "current_model" / "features"
    training_schema = load_csv(features_dir / "ml_training_schema.csv")
    fights_clean = load_csv(Path("data") / "current_model" / "clean" / "fights_clean.csv")
    if "event_name" not in training_schema.columns and not fights_clean.empty:
        training_schema = training_schema.merge(
            fights_clean[["fight_id", "event_name"]].drop_duplicates(),
            on="fight_id",
            how="left",
        )
    training_data = build_training_data(training_schema)
    split_frames = build_split_frames(training_data)
    db_path = Path("ufc_stats.db")

    save_csv(training_data, features_dir / "ml_training_data.csv")
    save_csv(split_frames.get("train", pd.DataFrame(columns=training_data.columns)), features_dir / "train.csv")
    save_csv(split_frames.get("validation", pd.DataFrame(columns=training_data.columns)), features_dir / "validation.csv")
    save_csv(split_frames.get("test", pd.DataFrame(columns=training_data.columns)), features_dir / "test.csv")

    with sqlite3.connect(str(db_path)) as conn:
        training_data.to_sql("ml_training_data", conn, if_exists="replace", index=False)
        split_frames.get("train", pd.DataFrame(columns=training_data.columns)).to_sql("ml_training_data_train", conn, if_exists="replace", index=False)
        split_frames.get("validation", pd.DataFrame(columns=training_data.columns)).to_sql("ml_training_data_validation", conn, if_exists="replace", index=False)
        split_frames.get("test", pd.DataFrame(columns=training_data.columns)).to_sql("ml_training_data_test", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
