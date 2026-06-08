"""
Centralized target-aware controller for Annex 96 / CityLearn Texas.

Goal:
    Control all 25 Texas buildings together so that portfolio net load
    tracks district_target.csv.

Uses:
    - harish_work/outputs/ml_datasets/tx_portfolio_ml_features.csv
    - harish_work/outputs/ml_datasets/tx_building_ml_features.csv
    - data/datasets/annex96_ce1_tx_neighborhood/district_target.csv

Important:
    This is not RL yet.
    This is a strong centralized predictive/target-aware controller.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from citylearn.citylearn import CityLearnEnv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "annex96_ce1_tx_neighborhood"
SCHEMA_FILE = DATASET_DIR / "schema.json"
DISTRICT_TARGET_FILE = DATASET_DIR / "district_target.csv"

ML_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
PORTFOLIO_FEATURE_FILE = ML_DIR / "tx_portfolio_ml_features.csv"
BUILDING_FEATURE_FILE = ML_DIR / "tx_building_ml_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "centralized_target_controller"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------

EVALUATION_DAYS = 90
STEPS_PER_DAY = 24
EVALUATION_STEPS = EVALUATION_DAYS * STEPS_PER_DAY

LOOKAHEAD_HOURS = 6

# Controller gains. These are the first values to tune.
BATTERY_GAIN = 2.00
COOLING_GAIN = 0.0
SOLAR_CHARGE_GAIN = 0.35
FUTURE_TARGET_GAIN = 0.25

MAX_BATTERY_ACTION = 1.0
MAX_COOLING_ACTION = 0.0

SOC_LOW = 0.20
SOC_HIGH = 0.85

COMFORT_MARGIN = 1.0

PLOT_STEPS = 168


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def choose_numeric_column(df: pd.DataFrame, preferred_names: list[str]) -> str:
    """Find a numeric column by preferred name or fallback to last numeric column."""

    lower_map = {c.lower(): c for c in df.columns}

    for name in preferred_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found.")

    return numeric_cols[-1]


def find_column_contains(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find first column whose lowercase name contains all keywords."""

    for col in df.columns:
        c = col.lower()
        if all(k.lower() in c for k in keywords):
            return col

    return None


def get_time_column(df: pd.DataFrame) -> str | None:
    for col in ["time_step", "step", "hour_index", "timestep"]:
        if col in df.columns:
            return col

    return None


def normalize_abs(value: float, reference: float, eps: float = 1e-6) -> float:
    return float(abs(value) / max(abs(reference), eps))


def safe_clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def env_reset(env: CityLearnEnv):
    result = env.reset()

    if isinstance(result, tuple):
        return result[0]

    return result


def env_step(env: CityLearnEnv, action):
    result = env.step(action)

    if len(result) == 5:
        observations, reward, terminated, truncated, info = result
        done = terminated or truncated
        return observations, reward, done, info

    observations, reward, done, info = result
    return observations, reward, done, info


# ---------------------------------------------------------------------
# Load target and feature data
# ---------------------------------------------------------------------

def load_district_target() -> pd.Series:
    df = pd.read_csv(DISTRICT_TARGET_FILE)

    target_col = choose_numeric_column(
        df,
        preferred_names=[
            "district_target",
            "target",
            "district_load_target",
            "load_target",
            "net_electricity_consumption",
        ],
    )

    return df[target_col].astype(float)


def find_first_active_target_step(target: pd.Series) -> int:
    active = np.where(np.abs(target.values) > 1e-6)[0]

    if len(active) == 0:
        raise ValueError("No active non-zero district target found.")

    return int(active[0])


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_df = pd.read_csv(PORTFOLIO_FEATURE_FILE)
    building_df = pd.read_csv(BUILDING_FEATURE_FILE)

    return portfolio_df, building_df


# ---------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------

def create_centralized_temp_schema(start_step: int, end_step: int) -> Path:
    with open(SCHEMA_FILE, "r") as f:
        schema = json.load(f)

    schema["root_directory"] = str(DATASET_DIR)
    schema["central_agent"] = True
    schema["simulation_start_time_step"] = int(start_step)
    schema["simulation_end_time_step"] = int(end_step)

    temp_dir = Path(tempfile.mkdtemp(prefix="citylearn_centralized_target_"))
    temp_schema_file = temp_dir / "schema_centralized_target.json"

    with open(temp_schema_file, "w") as f:
        json.dump(schema, f, indent=4)

    return temp_schema_file


# ---------------------------------------------------------------------
# Feature lookup
# ---------------------------------------------------------------------

class FeatureLookup:
    def __init__(
        self,
        portfolio_df: pd.DataFrame,
        building_df: pd.DataFrame,
        start_step: int,
    ):
        self.portfolio_df = portfolio_df.copy()
        self.building_df = building_df.copy()
        self.start_step = start_step

        self.portfolio_time_col = get_time_column(self.portfolio_df)
        self.building_time_col = get_time_column(self.building_df)

        self.portfolio_load_col = (
            find_column_contains(self.portfolio_df, ["portfolio", "net", "load"])
            or find_column_contains(self.portfolio_df, ["net", "load"])
            or find_column_contains(self.portfolio_df, ["electricity"])
        )

        self.portfolio_solar_col = (
            find_column_contains(self.portfolio_df, ["solar"])
            or find_column_contains(self.portfolio_df, ["pv"])
        )

        self.building_id_col = (
            find_column_contains(self.building_df, ["building", "id"])
            or find_column_contains(self.building_df, ["building", "index"])
        )

        self.soc_col = (
            find_column_contains(self.building_df, ["electrical", "storage", "soc"])
            or find_column_contains(self.building_df, ["battery", "soc"])
            or find_column_contains(self.building_df, ["soc"])
        )

        self.building_net_load_col = (
            find_column_contains(self.building_df, ["net", "load"])
            or find_column_contains(self.building_df, ["electricity"])
        )

        self.building_solar_col = (
            find_column_contains(self.building_df, ["solar"])
            or find_column_contains(self.building_df, ["pv"])
        )

        self.indoor_temp_col = find_column_contains(self.building_df, ["indoor", "temp"])
        self.cooling_setpoint_col = find_column_contains(self.building_df, ["cooling", "setpoint"])

        if self.portfolio_load_col is None:
            raise ValueError("Could not find portfolio net load column in portfolio ML features.")

    def get_portfolio_row(self, relative_step: int) -> pd.Series:
        absolute_step = self.start_step + relative_step

        if self.portfolio_time_col is not None:
            match = self.portfolio_df[self.portfolio_df[self.portfolio_time_col] == absolute_step]

            if match.empty:
                match = self.portfolio_df[self.portfolio_df[self.portfolio_time_col] == relative_step]

            if not match.empty:
                return match.iloc[0]

        idx = min(relative_step, len(self.portfolio_df) - 1)
        return self.portfolio_df.iloc[idx]

    def get_future_portfolio_rows(self, relative_step: int, horizon: int) -> pd.DataFrame:
        rows = []

        for k in range(horizon):
            idx = min(relative_step + k, len(self.portfolio_df) - 1)
            rows.append(self.get_portfolio_row(idx))

        return pd.DataFrame(rows)

    def get_building_rows(self, relative_step: int) -> pd.DataFrame:
        absolute_step = self.start_step + relative_step

        if self.building_time_col is not None:
            match = self.building_df[self.building_df[self.building_time_col] == absolute_step]

            if match.empty:
                match = self.building_df[self.building_df[self.building_time_col] == relative_step]

            if not match.empty:
                return match.copy()

        # Fallback: assume 25 buildings per time step.
        start = relative_step * 25
        end = start + 25
        return self.building_df.iloc[start:end].copy()

    def get_float(self, row: pd.Series, col: str | None, default: float = 0.0) -> float:
        if col is None:
            return default

        value = row.get(col, default)

        if pd.isna(value):
            return default

        return float(value)


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------

class CentralizedTargetController:
    def __init__(
        self,
        env: CityLearnEnv,
        target: pd.Series,
        lookup: FeatureLookup,
        start_step: int,
    ):
        self.env = env
        self.target = target
        self.lookup = lookup
        self.start_step = start_step

        self.action_names = env.action_names[0]
        self.action_space = env.action_space[0]

        self.n_actions = len(self.action_names)
        self.n_buildings = 25

        # Texas should normally have 2 active actions per building:
        # electrical_storage and cooling_device.
        self.actions_per_building = max(1, self.n_actions // self.n_buildings)

    def target_at(self, relative_step: int) -> float:
        absolute_step = self.start_step + relative_step

        if absolute_step < len(self.target):
            return float(self.target.iloc[absolute_step])

        return float(self.target.iloc[-1])

    def future_target_mean(self, relative_step: int, horizon: int) -> float:
        values = []

        for k in range(horizon):
            values.append(self.target_at(relative_step + k))

        return float(np.mean(values))

    def compute_building_weights(self, building_rows: pd.DataFrame) -> dict[str, np.ndarray]:
        n = self.n_buildings

        soc = np.full(n, 0.5)
        net_load = np.ones(n)
        solar = np.zeros(n)
        indoor = np.full(n, np.nan)
        cooling_sp = np.full(n, np.nan)

        for i in range(min(n, len(building_rows))):
            row = building_rows.iloc[i]

            soc[i] = self.lookup.get_float(row, self.lookup.soc_col, 0.5)
            net_load[i] = max(self.lookup.get_float(row, self.lookup.building_net_load_col, 1.0), 0.0)
            solar[i] = max(self.lookup.get_float(row, self.lookup.building_solar_col, 0.0), 0.0)
            indoor[i] = self.lookup.get_float(row, self.lookup.indoor_temp_col, np.nan)
            cooling_sp[i] = self.lookup.get_float(row, self.lookup.cooling_setpoint_col, np.nan)

        discharge_weight = np.clip(soc - SOC_LOW, 0.0, 1.0) * (net_load + 1e-3)
        charge_weight = np.clip(SOC_HIGH - soc, 0.0, 1.0) * (solar + 1e-3)

        if discharge_weight.sum() <= 1e-9:
            discharge_weight = np.ones(n)

        if charge_weight.sum() <= 1e-9:
            charge_weight = np.ones(n)

        discharge_weight = discharge_weight / discharge_weight.sum()
        charge_weight = charge_weight / charge_weight.sum()

        # Cooling permission:
        # If indoor temperature is safely below cooling setpoint + margin,
        # we can reduce cooling more aggressively.
        cooling_reduce_weight = np.ones(n)
        cooling_increase_weight = np.ones(n)

        for i in range(n):
            if not np.isnan(indoor[i]) and not np.isnan(cooling_sp[i]):
                if indoor[i] <= cooling_sp[i] + COMFORT_MARGIN:
                    cooling_reduce_weight[i] = 1.0
                else:
                    cooling_reduce_weight[i] = 0.2

                if indoor[i] >= cooling_sp[i]:
                    cooling_increase_weight[i] = 1.0
                else:
                    cooling_increase_weight[i] = 0.2

        cooling_reduce_weight = cooling_reduce_weight / cooling_reduce_weight.sum()
        cooling_increase_weight = cooling_increase_weight / cooling_increase_weight.sum()

        return {
            "soc": soc,
            "solar": solar,
            "discharge": discharge_weight,
            "charge": charge_weight,
            "cooling_reduce": cooling_reduce_weight,
            "cooling_increase": cooling_increase_weight,
        }

    def compute_actions(self, relative_step: int, live_portfolio_load: float | None = None) -> list[list[float]]:
        portfolio_row = self.lookup.get_portfolio_row(relative_step)
        building_rows = self.lookup.get_building_rows(relative_step)
        
        feature_predicted_load = self.lookup.get_float(
        portfolio_row,
        self.lookup.portfolio_load_col,
        default=0.0,)

        if live_portfolio_load is None:
            predicted_load = feature_predicted_load
        else:
            predicted_load = live_portfolio_load

        current_target = self.target_at(relative_step)
        current_error = current_target - predicted_load

        future_rows = self.lookup.get_future_portfolio_rows(relative_step, LOOKAHEAD_HOURS)
        future_predicted_load = future_rows[self.lookup.portfolio_load_col].astype(float).mean()
        future_target = self.future_target_mean(relative_step, LOOKAHEAD_HOURS)
        future_error = future_target - future_predicted_load

        solar_now = self.lookup.get_float(
            portfolio_row,
            self.lookup.portfolio_solar_col,
            default=0.0,
        )

        reference_load = max(abs(current_target), abs(predicted_load), 1.0)

        error_intensity = normalize_abs(current_error, reference_load)
        future_intensity = normalize_abs(future_error, reference_load)

        weights = self.compute_building_weights(building_rows)

        battery_actions = np.zeros(self.n_buildings)
        cooling_actions = np.zeros(self.n_buildings)

        # -----------------------------------------------------------------
        # Battery control: simple target-following version
        # -----------------------------------------------------------------
        battery_actions = np.zeros(self.n_buildings)

        if current_error > 0:
            # Live load is below target: increase load by charging batteries.
            charge_strength = min(BATTERY_GAIN * error_intensity, MAX_BATTERY_ACTION)
            battery_actions += charge_strength
        else:
            # Live load is above target: try to reduce load by discharging.
            discharge_strength = min(BATTERY_GAIN * error_intensity, MAX_BATTERY_ACTION)
            battery_actions -= discharge_strength
                

        # -----------------------------------------------------------------
        # Cooling control
        # -----------------------------------------------------------------
        # First stable version: keep cooling neutral.
        # We avoid aggressive cooling changes until live battery tracking works.
        cooling_actions = np.zeros(self.n_buildings)

        # Safety clip before mapping to CityLearn action vector.
        battery_actions = np.clip(battery_actions, -MAX_BATTERY_ACTION, MAX_BATTERY_ACTION)
        cooling_actions = np.clip(cooling_actions, -MAX_COOLING_ACTION, MAX_COOLING_ACTION)

        # -----------------------------------------------------------------
        # Convert to centralized CityLearn action vector
        # -----------------------------------------------------------------
        action_vector = []

        lows = np.array(self.action_space.low, dtype=float)
        highs = np.array(self.action_space.high, dtype=float)

        for action_index, action_name in enumerate(self.action_names):
            building_index = min(action_index // self.actions_per_building, self.n_buildings - 1)
            action_name_lower = action_name.lower()

            if "electrical_storage" in action_name_lower:
                value = battery_actions[building_index]
            elif "cooling_device" in action_name_lower:
                value = cooling_actions[building_index]
            else:
                value = 0.0

            value = safe_clip(value, lows[action_index], highs[action_index])
            action_vector.append(value)

        return [action_vector]


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------

def calculate_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual = np.array(actual, dtype=float)
    target = np.array(target, dtype=float)

    error = actual - target

    nmbe = 100.0 * np.mean(error) / max(np.mean(target), 1e-6)
    cv_rmse = 100.0 * math.sqrt(np.mean(error ** 2)) / max(np.mean(target), 1e-6)
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


def get_portfolio_load_from_features(
    lookup: FeatureLookup,
    relative_step: int,
) -> float:
    row = lookup.get_portfolio_row(relative_step)
    return lookup.get_float(row, lookup.portfolio_load_col, default=0.0)


def get_live_portfolio_load(env: CityLearnEnv) -> float:
    """
    Returns current live district net electricity consumption after env.step().
    CityLearnEnv.net_electricity_consumption is a time series list,
    so we must take the latest value, not sum the full list.
    """

    if hasattr(env, "net_electricity_consumption"):
        value = env.net_electricity_consumption

        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[-1])

        return float(value)

    loads = []

    for building in env.buildings:
        value = building.net_electricity_consumption

        if isinstance(value, (list, tuple, np.ndarray)):
            loads.append(float(value[-1]))
        else:
            loads.append(float(value))

    return float(np.sum(loads))


# ---------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------

def main() -> None:
    print("\n=== Centralized Target-Aware Controller ===")

    target = load_district_target()
    first_active_step = find_first_active_target_step(target)

    start_step = first_active_step
    end_step = start_step + EVALUATION_STEPS

    print(f"Start step: {start_step}")
    print(f"End step:   {end_step}")
    print(f"Steps:      {EVALUATION_STEPS}")

    portfolio_df, building_df = load_features()
    lookup = FeatureLookup(portfolio_df, building_df, start_step=start_step)

    temp_schema = create_centralized_temp_schema(start_step, end_step)
    env = CityLearnEnv(schema=str(temp_schema))

    print(f"Central agent: {env.central_agent}")
    print(f"Number of centralized actions: {len(env.action_names[0])}")
    print(f"Action names sample: {env.action_names[0][:6]}")

    controller = CentralizedTargetController(
        env=env,
        target=target,
        lookup=lookup,
        start_step=start_step,
    )

    observations = env_reset(env)

    records = []

    for relative_step in range(EVALUATION_STEPS):
        live_load_before_action = get_live_portfolio_load(env)
        actions = controller.compute_actions(
            relative_step,
            live_portfolio_load=live_load_before_action,)
        
        observations, reward, done, info = env_step(env, actions)
        

        # For now, actual portfolio load is taken from engineered portfolio features.
        # This makes the controller/evaluation aligned with your ML dataset.
        # Later, we can replace this with live CityLearn building net load extraction.
        actual_portfolio_load = get_live_portfolio_load(env)
        target_value = controller.target_at(relative_step)

        action_vector = actions[0]
        battery_actions = [
            value
            for name, value in zip(env.action_names[0], action_vector)
            if "electrical_storage" in name.lower()
        ]
        cooling_actions = [
            value
            for name, value in zip(env.action_names[0], action_vector)
            if "cooling_device" in name.lower()
        ]

        records.append(
            {
                "relative_step": relative_step,
                "absolute_step": start_step + relative_step,
                "district_target": target_value,
                "actual_portfolio_load": actual_portfolio_load,
                "tracking_error": actual_portfolio_load - target_value,
                "mean_battery_action": float(np.mean(battery_actions)) if battery_actions else 0.0,
                "mean_cooling_action": float(np.mean(cooling_actions)) if cooling_actions else 0.0,
                "min_battery_action": float(np.min(battery_actions)) if battery_actions else 0.0,
                "max_battery_action": float(np.max(battery_actions)) if battery_actions else 0.0,
                "min_cooling_action": float(np.min(cooling_actions)) if cooling_actions else 0.0,
                "max_cooling_action": float(np.max(cooling_actions)) if cooling_actions else 0.0,
                "reward": float(np.mean(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward),
            }
        )

        if done:
            break

    results = pd.DataFrame(records)

    metrics = calculate_metrics(
        actual=results["actual_portfolio_load"].values,
        target=results["district_target"].values,
    )

    results_file = OUTPUT_DIR / "centralized_target_controller_results.csv"
    metrics_file = OUTPUT_DIR / "centralized_target_controller_metrics.json"
    plot_file = OUTPUT_DIR / "centralized_target_controller_plot.png"

    results.to_csv(results_file, index=False)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== Metrics based on live CityLearn portfolio load ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved results: {results_file}")
    print(f"Saved metrics: {metrics_file}")

    # Plot
    plot_df = results.head(PLOT_STEPS)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["relative_step"], plot_df["district_target"], label="District Target")
    plt.plot(plot_df["relative_step"], plot_df["actual_portfolio_load"], label="Live CityLearn Portfolio Load")
    plt.xlabel("Hour")
    plt.ylabel("Portfolio load")
    plt.title("Centralized Target-Aware Controller: Target vs Feature Portfolio Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"Saved plot: {plot_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()