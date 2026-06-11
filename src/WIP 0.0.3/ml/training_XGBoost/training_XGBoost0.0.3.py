from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

try:
    # Optional plotting dependency. The script still runs if matplotlib is missing.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plotting
    plt = None


# Columns kept in the CSV for debugging / traceability but dropped before model training.
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
# A "low experience" fight is defined as the smaller of the two fighters' prior UFC fight counts.
LOW_EXPERIENCE_THRESHOLD = 3
# Where diagnostic plot files are saved if plotting is available.
ARTIFACT_DIR = Path("artifacts") / "xgboost"
FEATURES_DIR = Path("data") / "WIP 0.0.3" / "features"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load the prebuilt chronological splits. The model does not create its own split.
    required_files = [
        FEATURES_DIR / "train.csv",
        FEATURES_DIR / "validation.csv",
        FEATURES_DIR / "test.csv",
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing generated 0.0.3 feature files: "
            f"{missing_text}. Run src/WIP 0.0.3/ml/build_training_data0.0.3.py first."
        )

    train_df = pd.read_csv(FEATURES_DIR / "train.csv")
    validation_df = pd.read_csv(FEATURES_DIR / "validation.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test.csv")
    return train_df, validation_df, test_df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # y is the supervised target: 1 if fighter A won, 0 otherwise.
    y = df["fighter_a_won"]
    # X is every actual model feature after removing debug columns.
    x = df.drop(columns=DEBUG_DROP_COLUMNS)
    return x, y


def add_experience_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    # Pull pre-fight experience from fighter_snapshots so we can inspect low-experience bouts.
    snapshots = pd.read_csv(FEATURES_DIR / "fighter_snapshots.csv")
    snapshots = snapshots[["fight_id", "fighter_id", "ufc_fights"]].drop_duplicates()

    fighter_a_experience = snapshots.rename(
        columns={"fighter_id": "fighter_a_id", "ufc_fights": "fighter_a_prior_fights"}
    )
    fighter_b_experience = snapshots.rename(
        columns={"fighter_id": "fighter_b_id", "ufc_fights": "fighter_b_prior_fights"}
    )

    enriched = results_df.merge(fighter_a_experience, on=["fight_id", "fighter_a_id"], how="left")
    enriched = enriched.merge(fighter_b_experience, on=["fight_id", "fighter_b_id"], how="left")
    enriched["fighter_a_prior_fights"] = pd.to_numeric(enriched["fighter_a_prior_fights"], errors="coerce")
    enriched["fighter_b_prior_fights"] = pd.to_numeric(enriched["fighter_b_prior_fights"], errors="coerce")
    enriched["min_prior_fights"] = enriched[["fighter_a_prior_fights", "fighter_b_prior_fights"]].min(axis=1)
    return enriched


def add_weight_class_labels(results_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    # Recover a readable weight-class label from the one-hot encoded feature columns.
    weight_columns = [column for column in feature_df.columns if column.startswith("weight_class_")]
    if not weight_columns:
        results_df["weight_class_label"] = "unknown"
        return results_df

    weight_labels = (
        feature_df[weight_columns]
        .idxmax(axis=1)
        .str.replace("weight_class_", "", regex=False)
    )
    results_df["weight_class_label"] = weight_labels
    return results_df


def canonicalize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    # Keep one canonical row per real fight using the original schema orientation.
    schema = pd.read_csv(FEATURES_DIR / "ml_training_schema.csv")
    canonical_keys = schema[["fight_id", "fighter_a_id", "fighter_b_id", "fighter_a_won"]].drop_duplicates()
    canonical = results_df.merge(
        canonical_keys.assign(_is_canonical=1),
        on=["fight_id", "fighter_a_id", "fighter_b_id", "fighter_a_won"],
        how="left",
    )
    canonical = canonical[canonical["_is_canonical"] == 1].drop(columns="_is_canonical")
    return canonical.reset_index(drop=True)


def print_metric_block(label: str, y_true: pd.Series, probabilities: pd.Series, predictions: pd.Series) -> None:
    # Log loss uses probabilities, AUC measures ranking quality, and accuracy uses hard picks.
    print(f"{label} Log Loss:", log_loss(y_true, probabilities))
    print(f"{label} AUC:", roc_auc_score(y_true, probabilities))
    print(f"{label} Accuracy:", accuracy_score(y_true, predictions))


def summarize_feature_importance(model: XGBClassifier, feature_names: list[str]) -> pd.Series:
    # XGBoost exposes a relative importance score for each input feature.
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 20 Feature Importances")
    print(importance.head(20).to_string())
    return importance


def plot_feature_importance(importance: pd.Series) -> None:
    if plt is None:
        print("\nmatplotlib is not installed, skipping feature importance plot.")
        return

    # Save a top-20 horizontal bar chart so we can inspect what the model is leaning on most.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    top_importance = importance.head(20).sort_values()
    figure = plt.figure(figsize=(10, 7))
    top_importance.plot(kind="barh")
    plt.title("Top 20 XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path = ARTIFACT_DIR / "top_feature_importance_0.0.3.png"
    plt.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nSaved feature importance plot to {output_path}")


def inspect_prediction_behavior(results_df: pd.DataFrame) -> None:
    # "Confident wrong" examples are useful for spotting missing features or model blind spots.
    confident_wrong = results_df[results_df["pred_fighter_a_win"] != results_df["fighter_a_won"]].copy()
    confident_wrong["confidence"] = confident_wrong["prob_fighter_a_win"].where(
        confident_wrong["pred_fighter_a_win"] == 1,
        1 - confident_wrong["prob_fighter_a_win"],
    )

    closest_to_coin_flip = results_df.copy()
    # Near-0.5 probabilities are the fights the model is least certain about.
    closest_to_coin_flip["distance_from_50"] = (closest_to_coin_flip["prob_fighter_a_win"] - 0.5).abs()

    print("\nMost Confident Wrong Predictions")
    print(
        confident_wrong.sort_values("confidence", ascending=False)[
            [
                "event_date",
                "event_name",
                "fighter_a_name",
                "fighter_b_name",
                "fighter_a_won",
                "pred_fighter_a_win",
                "prob_fighter_a_win",
                "confidence",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nMost Uncertain Predictions")
    print(
        closest_to_coin_flip.sort_values("distance_from_50")[
            [
                "event_date",
                "event_name",
                "fighter_a_name",
                "fighter_b_name",
                "fighter_a_won",
                "pred_fighter_a_win",
                "prob_fighter_a_win",
                "distance_from_50",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


def subgroup_metrics(results_df: pd.DataFrame, label: str, mask: pd.Series) -> None:
    # Re-run the same validation metrics on a subset to find where the model is stronger/weaker.
    subset = results_df[mask].copy()
    if subset.empty:
        print(f"\n{label}: no rows")
        return

    subset_predictions = subset["pred_fighter_a_win"]
    subset_probabilities = subset["prob_fighter_a_win"]
    subset_target = subset["fighter_a_won"]

    print(f"\n{label} ({len(subset)} rows)")
    print("Log Loss:", log_loss(subset_target, subset_probabilities))
    print("AUC:", roc_auc_score(subset_target, subset_probabilities))
    print("Accuracy:", accuracy_score(subset_target, subset_predictions))


def weight_class_metrics(results_df: pd.DataFrame) -> None:
    # Weight-class slice to see whether model quality changes across divisions.
    print("\nValidation Metrics By Weight Class")
    for weight_class, subset in results_df.groupby("weight_class_label"):
        if len(subset) < 20:
            continue
        print(f"\n{weight_class} ({len(subset)} rows)")
        print("Log Loss:", log_loss(subset["fighter_a_won"], subset["prob_fighter_a_win"]))
        print("AUC:", roc_auc_score(subset["fighter_a_won"], subset["prob_fighter_a_win"]))
        print("Accuracy:", accuracy_score(subset["fighter_a_won"], subset["pred_fighter_a_win"]))


def main() -> None:
    # Use train for fitting and validation for model selection / diagnostics.
    train_df, validation_df, _ = load_data()

    x_train, y_train = split_features_and_target(train_df)
    x_val, y_val = split_features_and_target(validation_df)

    # Baseline XGBoost configuration. This is the actual model you were already training.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42,
    )

    # Early stopping watches validation log loss and stops if it stalls for 10 rounds.
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    # Probability of fighter A winning for each validation row.
    val_probability = model.predict_proba(x_val)[:, 1]
    # Convert probabilities into hard 0/1 picks using a 0.5 threshold.
    val_predictions = (val_probability >= 0.5).astype(int)

    print_metric_block("Validation", y_val, val_probability, val_predictions)

    # General feature-importance diagnostics.
    importance = summarize_feature_importance(model, x_train.columns.tolist())
    plot_feature_importance(importance)

    # Build a debug-friendly validation table containing predictions and fight metadata.
    validation_results = validation_df.copy()
    validation_results["prob_fighter_a_win"] = val_probability
    validation_results["pred_fighter_a_win"] = val_predictions
    validation_results["correct"] = (
        validation_results["pred_fighter_a_win"] == validation_results["fighter_a_won"]
    ).astype(int)
    validation_results = add_weight_class_labels(validation_results, x_val)
    validation_results = add_experience_columns(validation_results)
    canonical_validation_results = canonicalize_results(validation_results)

    print_metric_block(
        "Fight-Level Validation",
        canonical_validation_results["fighter_a_won"],
        canonical_validation_results["prob_fighter_a_win"],
        canonical_validation_results["pred_fighter_a_win"],
    )

    # General error inspection.
    inspect_prediction_behavior(canonical_validation_results)

    # Requested subgroup diagnostics.
    subgroup_metrics(
        canonical_validation_results,
        "Female Fights",
        canonical_validation_results["is_women_fight"] == 1,
    )
    subgroup_metrics(
        canonical_validation_results,
        "Title Fights",
        canonical_validation_results["is_title_fight"] == 1,
    )
    subgroup_metrics(
        canonical_validation_results,
        f"Low Experience Fights (min prior fights <= {LOW_EXPERIENCE_THRESHOLD})",
        canonical_validation_results["min_prior_fights"] <= LOW_EXPERIENCE_THRESHOLD,
    )
    weight_class_metrics(canonical_validation_results)


if __name__ == "__main__":
    main()
