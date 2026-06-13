#this file was used to test for which training parameters return the best results

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.current_model.training_XGBoost.training_XGBoost_validation import (
    LOW_EXPERIENCE_THRESHOLD,
    add_experience_columns,
    add_weight_class_labels,
    canonicalize_results,
    load_data,
    split_features_and_target,
)


ARTIFACT_DIR = Path("artifacts") / "xgboost_param_sweep"
BASELINE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 10,
    "random_state": 42,
}
PARAM_SWEEP = {
    "n_estimators": [150, 500, 800],
    "max_depth": [3, 5, 6],
    "learning_rate": [0.03, 0.07, 0.10],
    "subsample": [0.6, 0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.9, 1.0],
    "min_child_weight": [1, 5, 7],
    "gamma": [0.5, 1.0, 2.0],
    "reg_alpha": [0.1, 0.5, 1.0],
    "reg_lambda": [2.0, 5.0, 10.0],
}


def safe_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def metric_row(
    run_label: str,
    parameter_name: str,
    parameter_value: str,
    subgroup: str,
    subset: pd.DataFrame,
) -> dict[str, object]:
    return {
        "run_label": run_label,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "subgroup": subgroup,
        "rows": len(subset),
        "log_loss": float(log_loss(subset["fighter_a_won"], subset["prob_fighter_a_win"])),
        "auc": safe_auc(subset["fighter_a_won"], subset["prob_fighter_a_win"]),
        "accuracy": float(accuracy_score(subset["fighter_a_won"], subset["pred_fighter_a_win"])),
    }


def subgroup_frames(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {
        "overall": results_df,
        "female_fights": results_df[results_df["is_women_fight"] == 1],
        "title_fights": results_df[results_df["is_title_fight"] == 1],
        "low_experience": results_df[results_df["min_prior_fights"] <= LOW_EXPERIENCE_THRESHOLD],
    }
    for weight_class, subset in results_df.groupby("weight_class_label"):
        if len(subset) < 20:
            continue
        groups[f"weight_class:{weight_class}"] = subset
    return groups


def evaluate_run(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
    run_label: str,
    parameter_name: str,
    parameter_value: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    x_train, y_train = split_features_and_target(train_df)
    x_val, y_val = split_features_and_target(validation_df)

    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_probability = model.predict_proba(x_val)[:, 1]
    val_predictions = (val_probability >= 0.5).astype(int)

    validation_results = validation_df.copy()
    validation_results["prob_fighter_a_win"] = val_probability
    validation_results["pred_fighter_a_win"] = val_predictions
    validation_results = add_weight_class_labels(validation_results, x_val)
    validation_results = add_experience_columns(validation_results)
    validation_results = canonicalize_results(validation_results)

    overall = metric_row(run_label, parameter_name, parameter_value, "overall", validation_results)
    overall["best_iteration"] = getattr(model, "best_iteration", None)

    subgroup_rows = [
        metric_row(run_label, parameter_name, parameter_value, subgroup, subset)
        for subgroup, subset in subgroup_frames(validation_results).items()
        if not subset.empty
    ]
    return overall, subgroup_rows


def build_run_configs() -> list[tuple[str, str, str, dict[str, object]]]:
    configs: list[tuple[str, str, str, dict[str, object]]] = [
        ("baseline", "baseline", "baseline", BASELINE_PARAMS.copy())
    ]
    for parameter_name, values in PARAM_SWEEP.items():
        for value in values:
            params = BASELINE_PARAMS.copy()
            params[parameter_name] = value
            configs.append(
                (
                    f"{parameter_name}={value}",
                    parameter_name,
                    str(value),
                    params,
                )
            )
    return configs


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, _ = load_data()

    overall_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []

    for run_label, parameter_name, parameter_value, params in build_run_configs():
        print(f"Running {run_label}")
        overall, subgroup_metrics = evaluate_run(
            train_df=train_df,
            validation_df=validation_df,
            params=params,
            run_label=run_label,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        overall_rows.append(overall)
        subgroup_rows.extend(subgroup_metrics)

    overall_df = pd.DataFrame(overall_rows).sort_values(["parameter_name", "parameter_value"])
    subgroup_df = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "parameter_name", "parameter_value"])

    overall_path = ARTIFACT_DIR / "overall_results.csv"
    subgroup_path = ARTIFACT_DIR / "subgroup_results.csv"
    overall_df.to_csv(overall_path, index=False)
    subgroup_df.to_csv(subgroup_path, index=False)

    print(f"\nSaved overall results to {overall_path}")
    print(f"Saved subgroup results to {subgroup_path}")

    baseline_log_loss = float(
        overall_df.loc[overall_df["run_label"] == "baseline", "log_loss"].iloc[0]
    )
    winners = overall_df[overall_df["parameter_name"] != "baseline"].copy()
    winners["log_loss_delta_vs_baseline"] = winners["log_loss"] - baseline_log_loss
    winners = winners.sort_values(["log_loss", "auc", "accuracy"], ascending=[True, False, False])

    print("\nTop 10 runs by overall log loss")
    print(
        winners[
            [
                "run_label",
                "parameter_name",
                "parameter_value",
                "log_loss",
                "auc",
                "accuracy",
                "log_loss_delta_vs_baseline",
                "best_iteration",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
