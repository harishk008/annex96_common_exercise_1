from pathlib import Path

import pandas as pd
import wandb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "harish_work"
    / "outputs"
    / "ml_controller"
    / "tx_action_response_dataset.csv"
)

df = pd.read_csv(DATASET_FILE)

run = wandb.init(
    entity="CityLearn-TeamB",
    project="CityLearn",
    name="upload-action-response-dataset-with-charts",
    job_type="upload-dataset",
    config={
        "dataset": "annex96_ce1_tx_neighborhood",
        "rows": len(df),
        "columns": len(df.columns),
        "purpose": "ML response model training for predictive centralized controller",
    },
)

# 1. Log scalar summaries
wandb.log({
    "dataset/rows": len(df),
    "dataset/columns": len(df.columns),
    "summary/current_load_mean": df["current_portfolio_load"].mean(),
    "summary/next_load_mean": df["next_portfolio_load"].mean(),
    "summary/load_change_mean": df["load_change"].mean(),
    "summary/tracking_error_next_mae": df["tracking_error_next"].abs().mean(),
})

# 2. Log full dataset as a W&B Table
table = wandb.Table(dataframe=df)
wandb.log({"tables/action_response_dataset": table})

# 3. Log grouped summary tables
battery_summary = (
    df.groupby("battery_action")["next_portfolio_load"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
)

cooling_summary = (
    df.groupby("cooling_action")["next_portfolio_load"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
)

wandb.log({
    "tables/battery_response_summary": wandb.Table(dataframe=battery_summary),
    "tables/cooling_response_summary": wandb.Table(dataframe=cooling_summary),
})

# 4. Log ready-made W&B plots
wandb.log({
    "charts/next_load_vs_battery_action": wandb.plot.line(
        wandb.Table(dataframe=battery_summary),
        x="battery_action",
        y="mean",
        title="Mean Next Portfolio Load vs Battery Action",
    ),
    "charts/next_load_vs_cooling_action": wandb.plot.line(
        wandb.Table(dataframe=cooling_summary),
        x="cooling_action",
        y="mean",
        title="Mean Next Portfolio Load vs Cooling Action",
    ),
})

# 5. Log CSV as versioned artifact
artifact = wandb.Artifact(
    name="tx-action-response-dataset",
    type="dataset",
    description="CityLearn Texas action-response dataset for ML predictive centralized controller.",
    metadata={
        "rows": len(df),
        "columns": len(df.columns),
        "battery_actions": sorted(df["battery_action"].unique().tolist()),
        "cooling_actions": sorted(df["cooling_action"].unique().tolist()),
    },
)

artifact.add_file(str(DATASET_FILE))
run.log_artifact(artifact)

run.finish()

print("Uploaded dataset, tables, charts, and artifact to W&B.")