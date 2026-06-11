from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.build_training_data import build_split_frames, build_training_data
from src.utils.storage import load_csv, save_csv


def filter_non_debut_fights(training_schema: pd.DataFrame, fighter_snapshots: pd.DataFrame) -> pd.DataFrame:
    if training_schema.empty or fighter_snapshots.empty:
        return training_schema.iloc[0:0].copy()

    snapshots = fighter_snapshots.copy()
    snapshots["ufc_fights"] = pd.to_numeric(snapshots["ufc_fights"], errors="coerce")

    prior_fight_counts = (
        snapshots[["fight_id", "fighter_id", "ufc_fights"]]
        .drop_duplicates(subset=["fight_id", "fighter_id"])
    )

    eligible_fights = (
        prior_fight_counts.groupby("fight_id")["ufc_fights"]
        .apply(lambda values: bool((values > 0).all()) and len(values) == 2)
    )
    eligible_fight_ids = set(eligible_fights[eligible_fights].index.astype(str))

    filtered = training_schema[training_schema["fight_id"].astype(str).isin(eligible_fight_ids)].copy()
    return filtered.reset_index(drop=True)


def main() -> None:
    data_dir = Path("data")
    features_dir = data_dir / "features"
    variant_dir = features_dir / "features_0.2"

    training_schema = load_csv(features_dir / "ml_training_schema.csv")
    fighter_snapshots = load_csv(features_dir / "fighter_snapshots.csv")
    fights_clean = load_csv(data_dir / "clean" / "fights_clean.csv")

    if "event_name" not in training_schema.columns and not fights_clean.empty:
        training_schema = training_schema.merge(
            fights_clean[["fight_id", "event_name"]].drop_duplicates(),
            on="fight_id",
            how="left",
        )

    filtered_schema = filter_non_debut_fights(training_schema, fighter_snapshots)
    training_data = build_training_data(filtered_schema)
    split_frames = build_split_frames(training_data)

    save_csv(training_data, variant_dir / "ml_training_data.csv")
    save_csv(split_frames.get("train", pd.DataFrame(columns=training_data.columns)), variant_dir / "train.csv")
    save_csv(split_frames.get("validation", pd.DataFrame(columns=training_data.columns)), variant_dir / "validation.csv")
    save_csv(split_frames.get("test", pd.DataFrame(columns=training_data.columns)), variant_dir / "test.csv")


if __name__ == "__main__":
    main()
