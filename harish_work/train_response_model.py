from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import wandb

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "harish_work"
    / "outputs"
    / "ml_controller"
    / "tx_action_response_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_controller"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "tx_response_model_hist_gradient_boosting.joblib"
FEATURE_FILE = OUTPUT_DIR / "tx_response_model_features.json"
PREDICTION_FILE = OUTPUT_DIR / "tx_response_model_predictions.csv"


RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "hour",
    "day",
    "district_target",
    "current_portfolio_load",
    "battery_action",
    "cooling_action",
    "portfolio_net_load_feature",
    "outdoor_temp",
    "rolling_temp_3h",
    "sin_hour",
    "cos_hour",
    "portfolio_solar",
    "mean_soc",
    "mean_indoor_temp",
    "mean_cooling_setpoint",
    "mean_building_net_load",
]

TARGET_COLUMN = "next_portfolio_load"


def chronological_split(df: pd.DataFrame):
    """
    Split by time, not random.
    Train: first 70%
    Val: next 15%
    Test: last 15%
    """

    unique_steps = sorted(df["relative_step"].unique())
    n = len(unique_steps)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_steps = set(unique_steps[:train_end])
    val_steps = set(unique_steps[train_end:val_end])
    test_steps = set(unique_steps[val_end:])

    train_df = df[df["relative_step"].isin(train_steps)].copy()
    val_df = df[df["relative_step"].isin(val_steps)].copy()
    test_df = df[df["relative_step"].isin(test_steps)].copy()

    return train_df, val_df, test_df


def evaluate(model, df, split_name):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    pred = model.predict(x)

    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2 = r2_score(y, pred)

    metrics = {
        f"{split_name}/MAE": mae,
        f"{split_name}/RMSE": rmse,
        f"{split_name}/R2": r2,
    }

    pred_df = df[[
        "episode",
        "relative_step",
        "absolute_step",
        "district_target",
        "current_portfolio_load",
        "battery_action",
        "cooling_action",
        "next_portfolio_load",
    ]].copy()

    pred_df["predicted_next_portfolio_load"] = pred
    pred_df["prediction_error"] = pred_df["predicted_next_portfolio_load"] - pred_df["next_portfolio_load"]
    pred_df["split"] = split_name

    return metrics, pred_df


def main():
    print("\n=== Training ML response model ===")

    df = pd.read_csv(DATA_FILE)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    train_df, val_df, test_df = chronological_split(df)

    print(f"Rows total: {len(df)}")
    print(f"Rows train: {len(train_df)}")
    print(f"Rows val:   {len(val_df)}")
    print(f"Rows test:  {len(test_df)}")

    run = wandb.init(
        entity="CityLearn-TeamB",
        project="CityLearn",
        name="train-response-model-hist-gradient-boosting",
        job_type="train-model",
        config={
            "model": "HistGradientBoostingRegressor",
            "dataset": "tx_action_response_dataset.csv",
            "target": TARGET_COLUMN,
            "features": FEATURE_COLUMNS,
            "random_seed": RANDOM_SEED,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_SEED,
    )

    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    train_metrics, train_pred = evaluate(model, train_df, "train")
    val_metrics, val_pred = evaluate(model, val_df, "validation")
    test_metrics, test_pred = evaluate(model, test_df, "test")

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.log(metrics)

    all_predictions = pd.concat(
        [train_pred, val_pred, test_pred],
        ignore_index=True,
    )

    all_predictions.to_csv(PREDICTION_FILE, index=False)

    joblib.dump(model, MODEL_FILE)

    with open(FEATURE_FILE, "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
            },
            f,
            indent=4,
        )

    prediction_table = wandb.Table(dataframe=all_predictions.sample(
        min(5000, len(all_predictions)),
        random_state=RANDOM_SEED,
    ))

    wandb.log({
        "tables/predictions_sample": prediction_table,
        "summary/test_rmse": metrics["test/RMSE"],
        "summary/test_mae": metrics["test/MAE"],
        "summary/test_r2": metrics["test/R2"],
    })

    artifact = wandb.Artifact(
        name="tx-response-model-hist-gradient-boosting",
        type="model",
        description="ML response model predicting next CityLearn portfolio load from state and candidate actions.",
        metadata={
            "model": "HistGradientBoostingRegressor",
            "target": TARGET_COLUMN,
            "test_rmse": metrics["test/RMSE"],
            "test_mae": metrics["test/MAE"],
            "test_r2": metrics["test/R2"],
        },
    )

    artifact.add_file(str(MODEL_FILE))
    artifact.add_file(str(FEATURE_FILE))
    artifact.add_file(str(PREDICTION_FILE))

    run.log_artifact(artifact)
    run.finish()

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved features: {FEATURE_FILE}")
    print(f"Saved predictions: {PREDICTION_FILE}")


if __name__ == "__main__":
    main()