
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "harish_work"
    / "outputs"
    / "ml_controller"
    / "group_aware_controller"
)

RESULTS_FILE = OUTPUT_DIR / "tx_group_aware_ml_controller_results.csv"
METRICS_FILE = OUTPUT_DIR / "tx_group_aware_ml_controller_metrics.json"

# 3 days = 3 * 24 hourly steps = 72 hours
PLOT_STEPS = 72

PLOT_FILE = OUTPUT_DIR / "group_aware_controller_3day_plot.png"
PLOT_PDF_FILE = OUTPUT_DIR / "group_aware_controller_3day_plot.pdf"
THREE_DAY_METRICS_FILE = OUTPUT_DIR / "group_aware_controller_3day_metrics.json"


def compute_metrics(actual, target):
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)

    error = actual - target
    target_mean = np.mean(target)

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_abs_error = np.max(np.abs(error))

    nmbe_percent = 100.0 * np.sum(error) / (len(error) * target_mean)
    cv_rmse_percent = 100.0 * rmse / target_mean

    return {
        "NMBE_percent": float(nmbe_percent),
        "CV_RMSE_percent": float(cv_rmse_percent),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Max_Abs_Error": float(max_abs_error),
    }


def get_column(results, possible_names, label):
    for name in possible_names:
        if name in results.columns:
            return name

    raise ValueError(
        f"Could not find column for {label}.\n"
        f"Tried: {possible_names}\n"
        f"Available columns: {list(results.columns)}"
    )


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find results file:\n{RESULTS_FILE}\n\n"
            "Run the group-aware controller first to generate the CSV."
        )

    results = pd.read_csv(RESULTS_FILE)

    step_col = get_column(
        results,
        ["local_step", "relative_step", "absolute_step"],
        "time step",
    )

    target_col = get_column(
        results,
        ["district_target", "district_load_target", "TargetLoad"],
        "district load target",
    )

    actual_col = get_column(
        results,
        ["actual_portfolio_load", "portfolio_load", "current_portfolio_load"],
        "actual portfolio load",
    )

    predicted_col = None
    for col in ["predicted_next_load", "prediction", "ml_predicted_next_load"]:
        if col in results.columns:
            predicted_col = col
            break

    plot_df = results.head(PLOT_STEPS).copy()

    if len(plot_df) < PLOT_STEPS:
        print(
            f"Warning: requested {PLOT_STEPS} steps, "
            f"but only {len(plot_df)} rows are available."
        )

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df[step_col],
        plot_df[target_col],
        label="District Load Target",
        linewidth=2.4,
    )

    plt.plot(
        plot_df[step_col],
        plot_df[actual_col],
        label="Group-Aware Controller Load",
        linewidth=2.0,
    )

    if predicted_col is not None:
        plt.plot(
            plot_df[step_col],
            plot_df[predicted_col],
            label="Predicted Next Load",
            linewidth=1.6,
            alpha=0.75,
        )

    # Add day separators for readability
    start_step = int(plot_df[step_col].iloc[0])
    for offset in [24, 48, 72]:
        boundary = start_step + offset
        if boundary <= plot_df[step_col].max():
            plt.axvline(boundary, linestyle="--", linewidth=1, alpha=0.35)

    plt.xlabel("Hour")
    plt.ylabel("Portfolio load")
    plt.title("Group-Aware Controller Load Tracking — First 3 Days")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=300)
    plt.savefig(PLOT_PDF_FILE)
    plt.close()

    print(f"Saved PNG plot: {PLOT_FILE}")
    print(f"Saved PDF plot: {PLOT_PDF_FILE}")

    three_day_metrics = compute_metrics(
        actual=plot_df[actual_col],
        target=plot_df[target_col],
    )

    with open(THREE_DAY_METRICS_FILE, "w") as f:
        json.dump(three_day_metrics, f, indent=2)

    print("\n=== Group-Aware Controller Metrics: First 3 Days / 72 Hours ===")
    for key, value in three_day_metrics.items():
        print(f"{key}: {value:.4f}")

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            existing_metrics = json.load(f)

        print("\n=== Existing Full-Run Metrics File ===")
        for key, value in existing_metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

