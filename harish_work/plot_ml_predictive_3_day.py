
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
    / "ml_predictive_controller"
)

RESULTS_FILE = OUTPUT_DIR / "ml_predictive_controller_results.csv"
METRICS_FILE = OUTPUT_DIR / "ml_predictive_controller_metrics.json"

# 3 days = 3 * 24 hourly steps = 72 hours
PLOT_STEPS = 72

PLOT_FILE = OUTPUT_DIR / "ml_predictive_controller_3day_plot.png"
PLOT_PDF_FILE = OUTPUT_DIR / "ml_predictive_controller_3day_plot.pdf"
THREE_DAY_METRICS_FILE = OUTPUT_DIR / "ml_predictive_controller_3day_metrics.json"


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


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find results file:\n{RESULTS_FILE}\n\n"
            "Run the ML predictive controller first to generate the CSV."
        )

    results = pd.read_csv(RESULTS_FILE)

    required_cols = [
        "relative_step",
        "district_target",
        "actual_portfolio_load",
        "predicted_next_load",
    ]

    missing_cols = [col for col in required_cols if col not in results.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {RESULTS_FILE}: {missing_cols}\n"
            f"Available columns: {list(results.columns)}"
        )

    plot_df = results.head(PLOT_STEPS).copy()

    if len(plot_df) < PLOT_STEPS:
        print(
            f"Warning: requested {PLOT_STEPS} steps, "
            f"but only {len(plot_df)} rows are available."
        )

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df["relative_step"],
        plot_df["district_target"],
        label="District Load Target",
        linewidth=2.4,
    )

    plt.plot(
        plot_df["relative_step"],
        plot_df["actual_portfolio_load"],
        label="Actual Portfolio Load",
        linewidth=2.0,
    )

    plt.plot(
        plot_df["relative_step"],
        plot_df["predicted_next_load"],
        label="ML Predicted Next Load",
        linewidth=1.6,
        alpha=0.75,
    )

    # Add day separators for readability
    for day_boundary in [24, 48, 72]:
        if day_boundary <= plot_df["relative_step"].max():
            plt.axvline(day_boundary, linestyle="--", linewidth=1, alpha=0.35)

    plt.xlabel("Hour")
    plt.ylabel("Portfolio load")
    plt.title("ML Predictive Controller Load Tracking — First 3 Days")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=300)
    plt.savefig(PLOT_PDF_FILE)
    plt.close()

    print(f"Saved PNG plot: {PLOT_FILE}")
    print(f"Saved PDF plot: {PLOT_PDF_FILE}")

    three_day_metrics = compute_metrics(
        actual=plot_df["actual_portfolio_load"],
        target=plot_df["district_target"],
    )

    with open(THREE_DAY_METRICS_FILE, "w") as f:
        json.dump(three_day_metrics, f, indent=2)

    print("\n=== ML Predictive Controller Metrics: First 3 Days / 72 Hours ===")
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
