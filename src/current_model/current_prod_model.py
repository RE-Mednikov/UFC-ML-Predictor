from __future__ import annotations

# Run instructions:
# 1. Open PowerShell in the repo root:
#    C:\Users\Arie Mednikov\Documents\GitHub\UFC-ML-Predictor
# 2. Run:
#    .\.venv\Scripts\python.exe .\src\current_model\current_prod_model.py
# 3. When prompted, type the full name of fighter A exactly as it appears in your data.
# 4. Then type the full name of fighter B exactly as it appears in your data.
# 5. The script will ask for scheduled rounds (3 or 5), train the production
#    XGBoost model on train + validation + test, build both fighters' latest
#    stat snapshots from the clean historical data, and print the predicted
#    winner plus both win probabilities.
# 6. Press Ctrl+C at any time to stop the script.

import sys
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.build_training_data import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    normalize_weight_class,
    slugify_weight_class,
)
from src.current_model.build_training_schema import (  # noqa: E402
    build_fighter_snapshot,
    build_official_history_row,
    build_static_fighter_map,
    build_stats_lookup,
    build_training_row,
    update_elo,
)


DEBUG_DROP_COLUMNS = [
    "fighter_a_won",
    "fight_id",
    "event_id",
    "event_date",
    "event_name",
    "fighter_a_id",
    "fighter_a_name",
    "fighter_b_id",
    "fighter_b_name",
]
FEATURES_DIR = Path("data") / "current_model" / "features"
CLEAN_DIR = Path("data") / "current_model" / "clean"
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 7,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "eval_metric": "logloss",
    "random_state": 42,
}
SNAPSHOT_DISPLAY_COLUMNS = [
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


def load_training_frame() -> pd.DataFrame:
    train_df = pd.read_csv(FEATURES_DIR / "train.csv")
    validation_df = pd.read_csv(FEATURES_DIR / "validation.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test.csv")
    return pd.concat([train_df, validation_df, test_df], ignore_index=True)


def load_clean_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fights_clean = pd.read_csv(CLEAN_DIR / "fights_clean.csv")
    fighters_clean = pd.read_csv(CLEAN_DIR / "fighters_clean.csv")
    fighter_fight_stats_clean = pd.read_csv(CLEAN_DIR / "fighter_fight_stats_clean.csv")
    return fights_clean, fighters_clean, fighter_fight_stats_clean


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["fighter_a_won"]
    x = df.drop(columns=DEBUG_DROP_COLUMNS)
    return x, y


def train_model(training_df: pd.DataFrame) -> tuple[XGBClassifier, pd.DataFrame]:
    x_train, y_train = split_features_and_target(training_df)
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(x_train, y_train, verbose=False)
    return model, x_train


def build_name_lookup(fighters_clean: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    fighters = fighters_clean[["fighter_id", "fighter_name"]].dropna().copy()
    fighters["fighter_id"] = fighters["fighter_id"].astype(str)
    fighters["fighter_name"] = fighters["fighter_name"].astype(str).str.strip()
    fighters = fighters.drop_duplicates(subset=["fighter_id", "fighter_name"])
    latest_name_for_id = fighters.drop_duplicates(subset=["fighter_id"], keep="last")

    name_to_id: dict[str, str] = {}
    for _, row in latest_name_for_id.iterrows():
        name_to_id[row["fighter_name"].lower()] = row["fighter_id"]

    all_names = sorted(name_to_id.keys())
    return name_to_id, all_names


def resolve_fighter_id(input_name: str, name_to_id: dict[str, str], all_names: list[str]) -> tuple[str, str]:
    normalized_name = input_name.strip().lower()
    fighter_id = name_to_id.get(normalized_name)
    if fighter_id is None:
        close = get_close_matches(normalized_name, all_names, n=5, cutoff=0.65)
        if close:
            suggestions = ", ".join(sorted({name.title() for name in close}))
            raise ValueError(f"Fighter '{input_name}' was not found. Close matches: {suggestions}")
        raise ValueError(f"Fighter '{input_name}' was not found in fighters_clean.csv.")
    return fighter_id, normalized_name


def prepare_history_state(
    fights_clean: pd.DataFrame,
    fighter_fight_stats_clean: pd.DataFrame,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, float],
    pd.DataFrame,
]:
    stats_lookup = build_stats_lookup(fighter_fight_stats_clean)
    official_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stat_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    elo_ratings: dict[str, float] = defaultdict(lambda: 1500.0)

    fights = fights_clean.copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    fights = fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    for _, fight_row in fights.iterrows():
        fighter_a_id = str(fight_row["fighter_a_id"])
        fighter_b_id = str(fight_row["fighter_b_id"])
        is_decisive = (
            pd.notna(fight_row.get("fighter_a_won"))
            and fight_row.get("method_group") not in {"DRAW", "NC"}
            and pd.notna(fight_row.get("event_date"))
        )
        if not is_decisive:
            continue

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

    return official_history, stat_history, elo_ratings, fights


def latest_fighter_fights(fights_clean: pd.DataFrame, fighter_id: str) -> pd.DataFrame:
    fighter_fights = fights_clean[
        (fights_clean["fighter_a_id"].astype(str) == fighter_id)
        | (fights_clean["fighter_b_id"].astype(str) == fighter_id)
    ].copy()
    fighter_fights["event_date"] = pd.to_datetime(fighter_fights["event_date"], errors="coerce")
    return fighter_fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def infer_weight_class_and_rounds(
    fights_clean: pd.DataFrame,
    fighter_a_id: str,
    fighter_b_id: str,
) -> tuple[str, list[str]]:
    assumptions: list[str] = []
    fighter_a_fights = latest_fighter_fights(fights_clean, fighter_a_id)
    fighter_b_fights = latest_fighter_fights(fights_clean, fighter_b_id)

    if fighter_a_fights.empty or fighter_b_fights.empty:
        raise ValueError("One or both fighters do not have any recorded fights in fights_clean.csv.")

    fighter_a_latest = fighter_a_fights.iloc[-1]
    fighter_b_latest = fighter_b_fights.iloc[-1]

    fighter_a_class = normalize_weight_class(fighter_a_latest.get("weight_class"))
    fighter_b_class = normalize_weight_class(fighter_b_latest.get("weight_class"))

    common_classes = sorted(
        set(fighter_a_fights["weight_class"].map(normalize_weight_class))
        & set(fighter_b_fights["weight_class"].map(normalize_weight_class))
    )

    if fighter_a_class == fighter_b_class:
        chosen_class = fighter_a_class
        assumptions.append(
            f"Assumed weight class '{chosen_class}' because both fighters most recently competed there."
        )
    elif common_classes:
        chosen_class = common_classes[0]
        assumptions.append(
            f"Assumed weight class '{chosen_class}' because it is the first shared historical class in the data."
        )
    else:
        chosen_class = fighter_a_class
        assumptions.append(
            f"Fighters do not share a recorded weight class. Defaulted to fighter A's latest class '{chosen_class}'."
        )

    return chosen_class, assumptions


def prompt_scheduled_rounds() -> int:
    rounds_input = input("Enter scheduled rounds (3 or 5): ").strip()
    if rounds_input not in {"3", "5"}:
        raise ValueError("Scheduled rounds must be entered as 3 or 5.")
    return int(rounds_input)


def build_synthetic_fight_row(
    fights_clean: pd.DataFrame,
    fighter_a_id: str,
    fighter_a_name: str,
    fighter_b_id: str,
    fighter_b_name: str,
    weight_class: str,
    scheduled_rounds: int,
) -> pd.Series:
    latest_event_date = pd.to_datetime(fights_clean["event_date"], errors="coerce").max()
    synthetic_date = latest_event_date + pd.Timedelta(days=1)
    return pd.Series(
        {
            "fight_id": "prod_prediction",
            "event_id": "prod_prediction",
            "event_date": synthetic_date,
            "fighter_a_id": fighter_a_id,
            "fighter_a_name": fighter_a_name,
            "fighter_b_id": fighter_b_id,
            "fighter_b_name": fighter_b_name,
            "weight_class": weight_class,
            "scheduled_rounds": scheduled_rounds,
            "fighter_a_won": 0,
            "method_group": None,
            "fight_duration_seconds": None,
        }
    )


def build_prediction_features(
    feature_columns: list[str],
    base_training_row: dict[str, Any],
    weight_class: str,
    scheduled_rounds: int,
) -> pd.DataFrame:
    prediction_df = pd.DataFrame([{column: 0.0 for column in feature_columns}])

    prediction_df.at[0, "scheduled_rounds"] = scheduled_rounds
    prediction_df.at[0, "is_title_fight"] = int("title" in str(weight_class).lower())
    prediction_df.at[0, "is_women_fight"] = int("women" in str(weight_class).lower())

    for column in BASE_FEATURE_COLUMNS[1:]:
        if column in prediction_df.columns:
            prediction_df.at[0, column] = base_training_row.get(column)

    weight_column = f"weight_class_{slugify_weight_class(weight_class)}"
    if weight_column in prediction_df.columns:
        prediction_df.at[0, weight_column] = 1

    return prediction_df.apply(pd.to_numeric, errors="coerce")


def format_snapshot_table(snapshot_a: dict[str, Any], snapshot_b: dict[str, Any]) -> pd.DataFrame:
    fighter_a_df = pd.DataFrame(
        {
            "stat": SNAPSHOT_DISPLAY_COLUMNS,
            snapshot_a["fighter_name"]: [snapshot_a.get(column) for column in SNAPSHOT_DISPLAY_COLUMNS],
            snapshot_b["fighter_name"]: [snapshot_b.get(column) for column in SNAPSHOT_DISPLAY_COLUMNS],
        }
    )
    return fighter_a_df


def main() -> None:
    training_df = load_training_frame()
    model, training_features = train_model(training_df)

    fights_clean, fighters_clean, fighter_fight_stats_clean = load_clean_inputs()
    fighter_static = build_static_fighter_map(fighters_clean)
    name_to_id, all_names = build_name_lookup(fighters_clean)
    official_history, stat_history, elo_ratings, ordered_fights = prepare_history_state(
        fights_clean,
        fighter_fight_stats_clean,
    )

    fighter_a_input = input("Enter fighter A full name: ").strip()
    fighter_b_input = input("Enter fighter B full name: ").strip()
    if not fighter_a_input or not fighter_b_input:
        raise ValueError("Both fighter names are required.")

    fighter_a_id, fighter_a_name_key = resolve_fighter_id(fighter_a_input, name_to_id, all_names)
    fighter_b_id, fighter_b_name_key = resolve_fighter_id(fighter_b_input, name_to_id, all_names)
    if fighter_a_id == fighter_b_id:
        raise ValueError("Fighter A and fighter B must be different fighters.")

    fighter_a_name = fighter_static.get(fighter_a_id, {}).get("fighter_name") or fighter_a_name_key.title()
    fighter_b_name = fighter_static.get(fighter_b_id, {}).get("fighter_name") or fighter_b_name_key.title()

    scheduled_rounds = prompt_scheduled_rounds()
    weight_class, assumptions = infer_weight_class_and_rounds(
        ordered_fights,
        fighter_a_id,
        fighter_b_id,
    )
    assumptions.append(f"Used user-provided scheduled rounds: {scheduled_rounds}.")
    synthetic_fight_row = build_synthetic_fight_row(
        ordered_fights,
        fighter_a_id,
        fighter_a_name,
        fighter_b_id,
        fighter_b_name,
        weight_class,
        scheduled_rounds,
    )

    fighter_a_snapshot = build_fighter_snapshot(
        fighter_id=fighter_a_id,
        fighter_name=fighter_a_name,
        opponent_id=fighter_b_id,
        opponent_name=fighter_b_name,
        fight_row=synthetic_fight_row,
        fighter_static=fighter_static,
        official_history=official_history,
        stat_history=stat_history,
        elo_ratings=elo_ratings,
    )
    fighter_b_snapshot = build_fighter_snapshot(
        fighter_id=fighter_b_id,
        fighter_name=fighter_b_name,
        opponent_id=fighter_a_id,
        opponent_name=fighter_a_name,
        fight_row=synthetic_fight_row,
        fighter_static=fighter_static,
        official_history=official_history,
        stat_history=stat_history,
        elo_ratings=elo_ratings,
    )
    prediction_training_row = build_training_row(
        synthetic_fight_row,
        fighter_a_snapshot,
        fighter_b_snapshot,
    )
    prediction_features = build_prediction_features(
        feature_columns=training_features.columns.tolist(),
        base_training_row=prediction_training_row,
        weight_class=weight_class,
        scheduled_rounds=scheduled_rounds,
    )

    fighter_a_win_probability = float(model.predict_proba(prediction_features)[0, 1])
    fighter_b_win_probability = 1.0 - fighter_a_win_probability
    predicted_winner = fighter_a_name if fighter_a_win_probability >= 0.5 else fighter_b_name

    print(
        f"\nProduction model trained on {len(training_df)} rows from train + validation + test."
    )
    print("\nAssumptions")
    for assumption in assumptions:
        print(f"- {assumption}")

    print("\nGenerated Fighter Stat Line")
    print(format_snapshot_table(fighter_a_snapshot, fighter_b_snapshot).to_string(index=False))

    print("\nPrediction")
    print(f"Predicted winner: {predicted_winner}")
    print(f"{fighter_a_name} win probability: {fighter_a_win_probability:.4f}")
    print(f"{fighter_b_name} win probability: {fighter_b_win_probability:.4f}")


if __name__ == "__main__":
    main()
