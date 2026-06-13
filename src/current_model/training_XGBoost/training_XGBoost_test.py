from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.training_XGBoost.training_XGBoost_validation import (
    LOW_EXPERIENCE_THRESHOLD,
    add_experience_columns,
    add_weight_class_labels,
    canonicalize_results,
    inspect_prediction_behavior,
    load_data,
    split_features_and_target,
    subgroup_metrics,
    summarize_feature_importance,
    weight_class_metrics,
)

try:
    # Optional plotting dependency. The script still runs if matplotlib is missing.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plotting
    plt = None


# Save ready-model artifacts separately from validation-training artifacts.
ARTIFACT_DIR = Path("artifacts") / "xgboost_ready_model"


def print_metric_block(label: str, y_true: pd.Series, probabilities: pd.Series, predictions: pd.Series) -> None:
    # Log loss and Brier score use probabilities, AUC measures ranking quality, and accuracy uses hard picks.
    print(f"{label} Log Loss:", log_loss(y_true, probabilities))
    print(f"{label} Brier Score:", brier_score_loss(y_true, probabilities))
    print(f"{label} AUC:", roc_auc_score(y_true, probabilities))
    print(f"{label} Accuracy:", accuracy_score(y_true, predictions))


def plot_feature_importance(importance: pd.Series) -> None:
    if plt is None:
        print("\nmatplotlib is not installed, skipping feature importance plot.")
        return

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    top_importance = importance.head(20).sort_values()
    figure = plt.figure(figsize=(10, 7))
    top_importance.plot(kind="barh")
    plt.title("Top 20 XGBoost Feature Importances (Ready Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path = ARTIFACT_DIR / "top_feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nSaved feature importance plot to {output_path}")


def main() -> None:
    # Train the ready model on all pre-test data, then evaluate only on the held-out test split.
    train_df, validation_df, test_df = load_data()
    development_df = pd.concat([train_df, validation_df], ignore_index=True)

    x_development, y_development = split_features_and_target(development_df)
    x_test, y_test = split_features_and_target(test_df)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=7,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(x_development, y_development, verbose=False)

    test_probability = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probability >= 0.5).astype(int)

    print(
        f"Ready model trained on {len(development_df)} rows from train + validation "
        f"and evaluated on {len(test_df)} test rows."
    )
    print_metric_block("Test", y_test, test_probability, test_predictions)

    importance = summarize_feature_importance(model, x_development.columns.tolist())
    plot_feature_importance(importance)

    test_results = test_df.copy()
    test_results["prob_fighter_a_win"] = test_probability
    test_results["pred_fighter_a_win"] = test_predictions
    test_results["correct"] = (
        test_results["pred_fighter_a_win"] == test_results["fighter_a_won"]
    ).astype(int)
    test_results = add_weight_class_labels(test_results, x_test)
    test_results = add_experience_columns(test_results)
    canonical_test_results = canonicalize_results(test_results)

    print_metric_block(
        "Fight-Level Test",
        canonical_test_results["fighter_a_won"],
        canonical_test_results["prob_fighter_a_win"],
        canonical_test_results["pred_fighter_a_win"],
    )

    inspect_prediction_behavior(canonical_test_results)

    subgroup_metrics(
        canonical_test_results,
        "Female Fights",
        canonical_test_results["is_women_fight"] == 1,
    )
    subgroup_metrics(
        canonical_test_results,
        "Title Fights",
        canonical_test_results["is_title_fight"] == 1,
    )
    subgroup_metrics(
        canonical_test_results,
        f"Low Experience Fights (min prior fights <= {LOW_EXPERIENCE_THRESHOLD})",
        canonical_test_results["min_prior_fights"] <= LOW_EXPERIENCE_THRESHOLD,
    )
    weight_class_metrics(canonical_test_results)


if __name__ == "__main__":
    main()
