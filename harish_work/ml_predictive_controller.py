from pathlib import Path
import sys
import json
import math
import tempfile

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from citylearn.citylearn import CityLearnEnv


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "annex96_ce1_tx_neighborhood"
SCHEMA_FILE = DATASET_DIR / "schema.json"
DISTRICT_TARGET_FILE = DATASET_DIR / "district_target.csv"

ML_DATASET_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
PORTFOLIO_FEATURE_FILE = ML_DATASET_DIR / "tx_portfolio_ml_features.csv"
BUILDING_FEATURE_FILE = ML_DATASET_DIR / "tx_building_ml_features.csv"

ML_CONTROLLER_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_controller"
MODEL_FILE = ML_CONTROLLER_DIR / "tx_response_model_hist_gradient_boosting.joblib"
FEATURE_FILE = ML_CONTROLLER_DIR / "tx_response_model_features.json"

OUTPUT_DIR = ML_CONTROLLER_DIR / "ml_predictive_controller"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "ml_predictive_controller_results.csv"
METRICS_FILE = OUTPUT_DIR / "ml_predictive_controller_metrics.json"
PLOT_FILE = OUTPUT_DIR / "ml_predictive_controller_plot.png"


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

START_STEP = 3624
EVALUATION_DAYS = 90
STEPS_PER_DAY = 24
EVALUATION_STEPS = EVALUATION_DAYS * STEPS_PER_DAY

BATTERY_CANDIDATES = [-0.30, -0.15, 0.0, 0.15, 0.30]
COOLING_CANDIDATES = [0.0, 0.25, 0.50, 0.75, 1.0]

ACTION_SMOOTHNESS_WEIGHT = 8.0
TARGET_ERROR_WEIGHT = 1.0

PLOT_STEPS = 168
USE_WANDB = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def create_temp_schema():
    with open(SCHEMA_FILE, "r") as f:
        schema = json.load(f)

    schema["root_directory"] = str(DATASET_DIR)
    schema["central_agent"] = True
    schema["simulation_start_time_step"] = START_STEP
    schema["simulation_end_time_step"] = START_STEP + EVALUATION_STEPS

    temp_dir = Path(tempfile.mkdtemp(prefix="citylearn_ml_predictive_"))
    temp_schema = temp_dir / "schema.json"

    with open(temp_schema, "w") as f:
        json.dump(schema, f, indent=4)

    return temp_schema


def env_reset(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def env_step(env, action):
    result = env.step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = result
    return obs, reward, done, info


def get_live_portfolio_load(env):
    value = env.net_electricity_consumption

    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[-1])

    return float(value)


def make_uniform_action(env, battery_value, cooling_value):
    action_names = env.action_names[0]
    lows = env.action_space[0].low
    highs = env.action_space[0].high

    action = []

    for i, name in enumerate(action_names):
        name_lower = name.lower()

        if "electrical_storage" in name_lower:
            value = battery_value
        elif "cooling_device" in name_lower:
            value = cooling_value
        else:
            value = 0.0

        value = float(np.clip(value, lows[i], highs[i]))
        action.append(value)

    return [action]


def load_target():
    df = pd.read_csv(DISTRICT_TARGET_FILE)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric column found in district_target.csv")

    target_col = numeric_cols[-1]
    return df[target_col].astype(float)


def get_time_column(df):
    for col in ["time_step", "step", "hour_index", "timestep"]:
        if col in df.columns:
            return col

    return None


def get_row_by_step(df, time_col, relative_step):
    absolute_step = START_STEP + relative_step

    if time_col is not None:
        match = df[df[time_col] == absolute_step]

        if match.empty:
            match = df[df[time_col] == relative_step]

        if not match.empty:
            return match.iloc[0]

    idx = min(relative_step, len(df) - 1)
    return df.iloc[idx]


def get_building_rows_by_step(df, time_col, relative_step):
    absolute_step = START_STEP + relative_step

    if time_col is not None:
        match = df[df[time_col] == absolute_step]

        if match.empty:
            match = df[df[time_col] == relative_step]

        if not match.empty:
            return match

    start = relative_step * 25
    end = start + 25
    return df.iloc[start:end]


def find_col_contains(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if all(k.lower() in col_lower for k in keywords):
            return col

    return None


def safe_get(row, col, default=0.0):
    if col is None:
        return default

    value = row.get(col, default)

    if pd.isna(value):
        return default

    return float(value)


def calculate_metrics(actual, target):
    actual = np.array(actual, dtype=float)
    target = np.array(target, dtype=float)

    error = actual - target
    denominator = max(abs(np.mean(target)), 1e-6)

    nmbe = 100.0 * np.mean(error) / denominator
    cv_rmse = 100.0 * math.sqrt(np.mean(error ** 2)) / denominator
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(error ** 2)))
    max_abs_error = float(np.max(np.abs(error)))

    return {
        "NMBE_percent": float(nmbe),
        "CV_RMSE_percent": float(cv_rmse),
        "MAE": mae,
        "RMSE": rmse,
        "Max_Abs_Error": max_abs_error,
    }
    
def get_live_observation_summary(env):
    """
    Extract useful live state values from CityLearn observations.
    Works with central_agent=True.
    """

    obs_names = env.observation_names[0]
    obs_values = env.observations[0]

    soc_values = []
    indoor_temp_values = []
    cooling_setpoint_values = []
    net_load_values = []

    for name, value in zip(obs_names, obs_values):
        name_lower = name.lower()

        if "electrical_storage_soc" in name_lower:
            soc_values.append(float(value))

        elif "indoor_dry_bulb_temperature" in name_lower or "indoor_temp" in name_lower:
            indoor_temp_values.append(float(value))

        elif "cooling_set_point" in name_lower or "cooling_setpoint" in name_lower:
            cooling_setpoint_values.append(float(value))

        elif "net_electricity_consumption" in name_lower:
            net_load_values.append(float(value))

    return {
        "mean_soc": float(np.mean(soc_values)) if soc_values else 0.0,
        "mean_indoor_temp": float(np.mean(indoor_temp_values)) if indoor_temp_values else 0.0,
        "mean_cooling_setpoint": float(np.mean(cooling_setpoint_values)) if cooling_setpoint_values else 0.0,
        "mean_building_net_load": float(np.mean(net_load_values)) if net_load_values else 0.0,
    }

# ------------------------------------------------------------
# Feature builder
# ------------------------------------------------------------

class FeatureBuilder:
    def __init__(self, portfolio_df, building_df, feature_columns):
        self.portfolio_df = portfolio_df
        self.building_df = building_df
        self.feature_columns = feature_columns

        self.portfolio_time_col = get_time_column(portfolio_df)
        self.building_time_col = get_time_column(building_df)

        self.portfolio_cols = {
            "portfolio_net_load_feature": (
                find_col_contains(portfolio_df, ["portfolio", "net", "load"])
                or find_col_contains(portfolio_df, ["net", "load"])
            ),
            "outdoor_temp": find_col_contains(portfolio_df, ["outdoor", "temp"]),
            "rolling_temp_3h": find_col_contains(portfolio_df, ["rolling", "temp"]),
            "sin_hour": find_col_contains(portfolio_df, ["sin", "hour"]),
            "cos_hour": find_col_contains(portfolio_df, ["cos", "hour"]),
            "portfolio_solar": (
                find_col_contains(portfolio_df, ["solar"])
                or find_col_contains(portfolio_df, ["pv"])
            ),
        }

        self.building_cols = {
            "mean_soc": (
                find_col_contains(building_df, ["soc"])
                or find_col_contains(building_df, ["storage"])
            ),
            "mean_indoor_temp": find_col_contains(building_df, ["indoor", "temp"]),
            "mean_cooling_setpoint": find_col_contains(building_df, ["cooling", "setpoint"]),
            "mean_building_net_load": (
                find_col_contains(building_df, ["net", "load"])
                or find_col_contains(building_df, ["electricity"])
            ),
        }

    def build_row(
        self,
        relative_step,
        district_target,
        current_portfolio_load,
        battery_action,
        cooling_action,
        live_summary= None
    ):
        portfolio_row = get_row_by_step(
            self.portfolio_df,
            self.portfolio_time_col,
            relative_step,
        )

        building_rows = get_building_rows_by_step(
            self.building_df,
            self.building_time_col,
            relative_step,
        )

        row = {
            "hour": relative_step % 24,
            "day": relative_step // 24,
            "district_target": district_target,
            "current_portfolio_load": current_portfolio_load,
            "battery_action": battery_action,
            "cooling_action": cooling_action,
        }

        for out_name, col in self.portfolio_cols.items():
            row[out_name] = safe_get(portfolio_row, col, default=0.0)

        if live_summary is not None:
            row["mean_soc"] = live_summary.get("mean_soc", 0.0)
            row["mean_indoor_temp"] = live_summary.get("mean_indoor_temp", 0.0)
            row["mean_cooling_setpoint"] = live_summary.get("mean_cooling_setpoint", 0.0)
            row["mean_building_net_load"] = live_summary.get("mean_building_net_load", 0.0)
        else:
            for out_name, col in self.building_cols.items():
                if col is not None and col in building_rows.columns:
                    row[out_name] = float(building_rows[col].mean())
                else:
                    row[out_name] = 0.0

        # Ensure every model feature exists.
        final_row = {}

        for col in self.feature_columns:
            final_row[col] = row.get(col, 0.0)

        return final_row


# ------------------------------------------------------------
# Controller logic
# ------------------------------------------------------------

def choose_best_action(
    model,
    feature_builder,
    relative_step,
    district_target,
    current_portfolio_load,
    previous_battery_action,
    previous_cooling_action,
    prediction_bias,
    live_summary,
    
):
    candidate_rows = []
    candidate_info = []

    for battery_action in BATTERY_CANDIDATES:
        for cooling_action in COOLING_CANDIDATES:
            row = feature_builder.build_row(
                relative_step=relative_step,
                district_target=district_target,
                current_portfolio_load=current_portfolio_load,
                battery_action=battery_action,
                cooling_action=cooling_action,
                live_summary = live_summary,
                
            )

            candidate_rows.append(row)
            candidate_info.append((battery_action, cooling_action))

    candidate_df = pd.DataFrame(candidate_rows)
    predicted_loads = model.predict(candidate_df[feature_builder.feature_columns]) + prediction_bias

    best_score = float("inf")
    best_index = 0

    for i, predicted_load in enumerate(predicted_loads):
        battery_action, cooling_action = candidate_info[i]

        target_error_score = abs(predicted_load - district_target)

        smoothness_score = (
            abs(battery_action - previous_battery_action)
            + abs(cooling_action - previous_cooling_action)
        )

        score = (
            TARGET_ERROR_WEIGHT * target_error_score
            + ACTION_SMOOTHNESS_WEIGHT * smoothness_score
        )

        if score < best_score:
            best_score = score
            best_index = i

    best_battery_action, best_cooling_action = candidate_info[best_index]
    best_predicted_load = float(predicted_loads[best_index])

    return {
        "battery_action": best_battery_action,
        "cooling_action": best_cooling_action,
        "predicted_next_load": best_predicted_load,
        "predicted_tracking_error": best_predicted_load - district_target,
        "score": best_score,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== ML Predictive Controller ===")

    model = joblib.load(MODEL_FILE)

    with open(FEATURE_FILE, "r") as f:
        feature_info = json.load(f)

    feature_columns = feature_info["feature_columns"]

    target = load_target()
    portfolio_df = pd.read_csv(PORTFOLIO_FEATURE_FILE)
    building_df = pd.read_csv(BUILDING_FEATURE_FILE)

    feature_builder = FeatureBuilder(
        portfolio_df=portfolio_df,
        building_df=building_df,
        feature_columns=feature_columns,
    )

    env = CityLearnEnv(schema=str(create_temp_schema()))
    env_reset(env)

    print(f"Central agent: {env.central_agent}")
    print(f"Number of actions: {len(env.action_names[0])}")
    print(f"Action names sample: {env.action_names[0][:6]}")

    run = None

    if USE_WANDB:
        run = wandb.init(
            entity="CityLearn-TeamB",
            project="CityLearn",
            name="ml-predictive-controller",
            job_type="evaluate-controller",
            config={
                "controller": "ML predictive controller",
                "dataset": "annex96_ce1_tx_neighborhood",
                "start_step": START_STEP,
                "evaluation_steps": EVALUATION_STEPS,
                "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
                "battery_candidates": BATTERY_CANDIDATES,
                "cooling_candidates": COOLING_CANDIDATES,
                "action_smoothness_weight": ACTION_SMOOTHNESS_WEIGHT,
                "target_error_weight": TARGET_ERROR_WEIGHT,
            },
        )

    records = []

    previous_battery_action = 0.0
    previous_cooling_action = 0.0
    prediction_bias = 0.0
    BIAS_LEARNING_RATE = 0.10

    for relative_step in range(EVALUATION_STEPS):
        absolute_step = START_STEP + relative_step

        district_target = float(target.iloc[absolute_step]) if absolute_step < len(target) else float(target.iloc[-1])
        current_load = get_live_portfolio_load(env)
        live_summary = get_live_observation_summary(env)

        decision = choose_best_action(
            model=model,
            feature_builder=feature_builder,
            relative_step=relative_step,
            district_target=district_target,
            current_portfolio_load=current_load,
            previous_battery_action=previous_battery_action,
            previous_cooling_action=previous_cooling_action,
            prediction_bias = prediction_bias,
            live_summary=live_summary,
        )

        battery_action = decision["battery_action"]
        cooling_action = decision["cooling_action"]

        action = make_uniform_action(
            env=env,
            battery_value=battery_action,
            cooling_value=cooling_action,
        )

        obs, reward, done, info = env_step(env, action)

        actual_load = get_live_portfolio_load(env)
        raw_prediction_error = actual_load - decision["predicted_next_load"]
        prediction_bias = (
        (1.0 - BIAS_LEARNING_RATE) * prediction_bias
        + BIAS_LEARNING_RATE * raw_prediction_error)
        tracking_error = actual_load - district_target

        record = {
            "relative_step": relative_step,
            "absolute_step": absolute_step,
            "district_target": district_target,
            "current_portfolio_load_before_action": current_load,
            "actual_portfolio_load": actual_load,
            "tracking_error": tracking_error,
            "abs_tracking_error": abs(tracking_error),
            "battery_action": battery_action,
            "cooling_action": cooling_action,
            "predicted_next_load": decision["predicted_next_load"],
            "predicted_tracking_error": decision["predicted_tracking_error"],
            "prediction_error_vs_actual": decision["predicted_next_load"] - actual_load,
            "decision_score": decision["score"],
            "reward": float(np.mean(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward),
            "prediction_bias": prediction_bias,
            "mean_soc_live": live_summary["mean_soc"],
            "mean_indoor_temp_live": live_summary["mean_indoor_temp"],
            "mean_cooling_setpoint_live": live_summary["mean_cooling_setpoint"],
            "mean_building_net_load_live": live_summary["mean_building_net_load"],
        }

        records.append(record)

        if USE_WANDB:
            wandb.log({
                "step": relative_step,
                "load/district_target": district_target,
                "load/current_before_action": current_load,
                "load/actual_after_action": actual_load,
                "load/predicted_next_load": decision["predicted_next_load"],
                "error/tracking_error": tracking_error,
                "error/abs_tracking_error": abs(tracking_error),
                "error/prediction_error_vs_actual": decision["predicted_next_load"] - actual_load,
                "action/battery_action": battery_action,
                "action/cooling_action": cooling_action,
                "controller/decision_score": decision["score"],
            })

        previous_battery_action = battery_action
        previous_cooling_action = cooling_action

        if relative_step % 240 == 0:
            print(
                f"step {relative_step:4d} | "
                f"target {district_target:8.2f} | "
                f"current {current_load:8.2f} | "
                f"pred {decision['predicted_next_load']:8.2f} | "
                f"actual {actual_load:8.2f} | "
                f"battery {battery_action:+.2f} | "
                f"cooling {cooling_action:.2f}"
            )

        if done:
            break

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_FILE, index=False)

    metrics = calculate_metrics(
        actual=results["actual_portfolio_load"].values,
        target=results["district_target"].values,
    )

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== ML Predictive Controller Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    plot_df = results.head(PLOT_STEPS)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["relative_step"], plot_df["district_target"], label="District Target")
    plt.plot(plot_df["relative_step"], plot_df["actual_portfolio_load"], label="Actual Portfolio Load")
    plt.plot(plot_df["relative_step"], plot_df["predicted_next_load"], label="ML Predicted Load", alpha=0.7)
    plt.xlabel("Hour")
    plt.ylabel("Portfolio load")
    plt.title("ML Predictive Controller: Target vs Actual Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()

    if USE_WANDB:
        wandb.log({
            "metrics/NMBE_percent": metrics["NMBE_percent"],
            "metrics/CV_RMSE_percent": metrics["CV_RMSE_percent"],
            "metrics/MAE": metrics["MAE"],
            "metrics/RMSE": metrics["RMSE"],
            "metrics/Max_Abs_Error": metrics["Max_Abs_Error"],
            "tables/controller_results_sample": wandb.Table(
                dataframe=results.sample(min(5000, len(results)), random_state=42)
            ),
        })

        artifact = wandb.Artifact(
            name="ml-predictive-controller-results",
            type="controller-results",
            description="Results from ML predictive centralized CityLearn controller.",
            metadata=metrics,
        )

        artifact.add_file(str(RESULTS_FILE))
        artifact.add_file(str(METRICS_FILE))
        artifact.add_file(str(PLOT_FILE))

        run.log_artifact(artifact)
        run.finish()

    print(f"\nSaved results: {RESULTS_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved plot: {PLOT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()