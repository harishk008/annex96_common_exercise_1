import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# Path setup
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
HARISH_WORK_DIR = REPO_ROOT / "harish_work"
OUTPUT_DIR = HARISH_WORK_DIR / "outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(HARISH_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(HARISH_WORK_DIR))

from citylearn.citylearn import CityLearnEnv
from feature_engineer import FeatureEngineer
from rule_based_controller import RuleBasedTexasController


# ==========================================
# Configuration
# ==========================================
DATASET_NAME = "annex96_ce1_tx_neighborhood"
DATASET_DIR = REPO_ROOT / "data" / "datasets" / DATASET_NAME

SCHEMA_PATH = DATASET_DIR / "schema.json"
DISTRICT_TARGET_PATH = DATASET_DIR / "district_target.csv"

EVALUATION_DAYS = 90
EVALUATION_STEPS = EVALUATION_DAYS * 24 # 90 days × 24 hourly steps/day
TARGET_EPSILON = 1e-6       # threshold for detecting active target period


def load_district_target():
    """
    Load the real Texas district target profile from district_target.csv.
    """
    target_df = pd.read_csv(DISTRICT_TARGET_PATH)

    if "district_load_target" not in target_df.columns:
        raise ValueError(
            f"'district_load_target' column not found in {DISTRICT_TARGET_PATH}"
        )

    return target_df["district_load_target"].astype(float).to_numpy()


def find_first_active_target_step(target_series):
    """
    Find the first time step where the district target is non-zero.
    """
    active_indices = np.where(np.abs(target_series) > TARGET_EPSILON)[0]

    if len(active_indices) == 0:
        raise ValueError("No non-zero district target found in district_target.csv.")

    return int(active_indices[0])


def create_active_period_schema(target_start_step, evaluation_steps):
    """
    Create a temporary schema for the active district target period.

    This script applies a controller action and then logs the resulting
    observation. Therefore, the environment starts one step before the
    active target window.
    """

    env_start_step = max(int(target_start_step) - 1, 0)
    env_end_step = int(target_start_step + evaluation_steps - 1)

    with open(SCHEMA_PATH, "r") as file:
        schema = json.load(file)

    schema["simulation_start_time_step"] = env_start_step
    schema["simulation_end_time_step"] = env_end_step

    # Critical:
    # The temporary schema is saved in harish_work/outputs, but the real
    # dataset CSV files are inside DATASET_DIR.
    schema["root_directory"] = str(DATASET_DIR)

    active_schema_path = OUTPUT_DIR / "tx_real_baseline_schema.json"

    with open(active_schema_path, "w") as file:
        json.dump(schema, file, indent=4)

    return active_schema_path, env_start_step, env_end_step


def is_done(terminated, truncated):
    """
    Safely handle bool or list-like termination flags.
    """
    return bool(np.any(terminated)) or bool(np.any(truncated))


def get_load_index(observation_names):
    """
    Find net electricity consumption index.
    """
    return observation_names.index("net_electricity_consumption")


def get_action_value(action_names, action_array, action_name):
    """
    Read one action value by action name.
    """
    if action_name not in action_names:
        return np.nan

    index = action_names.index(action_name)
    return action_array[index]


def calculate_metrics(actual, target):
    """
    Calculate NMBE and CV-RMSE.
    """
    valid_mask = np.abs(target) > TARGET_EPSILON

    actual_valid = actual[valid_mask]
    target_valid = target[valid_mask]

    if len(target_valid) == 0:
        return np.nan, np.nan

    mean_target = np.mean(target_valid)

    if abs(mean_target) <= TARGET_EPSILON:
        return np.nan, np.nan

    nmbe = (np.mean(actual_valid - target_valid) / mean_target) * 100
    cv_rmse = (
        np.sqrt(np.mean((actual_valid - target_valid) ** 2)) / mean_target
    ) * 100

    return nmbe, cv_rmse


def extract_real_baseline():
    print("=" * 70)
    print("Texas Real Baseline Extraction")
    print("=" * 70)

    print(f"Original schema path: {SCHEMA_PATH}")
    print(f"District target path: {DISTRICT_TARGET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Load district target
    # ==========================================
    district_target_series = load_district_target()
    print(f"\nLoaded district target profile with {len(district_target_series)} time steps.")

    target_start_step = find_first_active_target_step(district_target_series)
    target_end_step = target_start_step + EVALUATION_STEPS - 1

    print(f"First active target step: {target_start_step}")
    print(f"Baseline extraction window: step {target_start_step} to step {target_end_step}")

    if target_end_step >= len(district_target_series):
        raise ValueError(
            "Baseline extraction window is longer than available district target profile."
        )

    # ==========================================
    # Create temporary active-period schema
    # ==========================================
    active_schema_path, env_start_step, env_end_step = create_active_period_schema(
        target_start_step=target_start_step,
        evaluation_steps=EVALUATION_STEPS,
    )

    print(f"Temporary baseline schema: {active_schema_path}")
    print(f"Environment window: step {env_start_step} to step {env_end_step}")

    # ==========================================
    # Create environment
    # ==========================================
    env = CityLearnEnv(schema=str(active_schema_path))
    observations, _ = env.reset()

    number_of_buildings = len(env.action_space)

    print(f"\nNumber of buildings: {number_of_buildings}")
    print(f"Action names for Building 1: {env.action_names[0]}")
    print(f"Number of observations for Building 1: {len(env.observation_names[0])}")

    # ==========================================
    # Create controller and feature engineers
    # ==========================================
    controller = RuleBasedTexasController(
        action_space=env.action_space,
        observation_names=env.observation_names,
        action_names=env.action_names,
    )

    engineers = [
        FeatureEngineer(env.observation_names[i])
        for i in range(number_of_buildings)
    ]

    load_indices = [
        get_load_index(env.observation_names[i])
        for i in range(number_of_buildings)
    ]

    # ==========================================
    # Storage
    # ==========================================
    building_rows = []
    portfolio_rows = []

    done = False
    logged_step = 0

    print(f"\nRunning rule-based baseline for {EVALUATION_STEPS} hourly steps...")

    while not done and logged_step < EVALUATION_STEPS:
        absolute_step = target_start_step + logged_step
        district_target = district_target_series[absolute_step]

        # Controller acts based on current observations.
        actions = controller.predict(observations)

        # Step environment.
        observations, rewards, terminated, truncated, info = env.step(actions)

        # ==========================================
        # Building-level rows after action is applied
        # ==========================================
        current_step_building_features = []

        for building_index in range(number_of_buildings):
            building_obs = observations[building_index]
            engineered_features = engineers[building_index].process_observation(
                building_obs
            )

            action_names = env.action_names[building_index]
            action_array = actions[building_index]

            battery_action = get_action_value(
                action_names,
                action_array,
                "electrical_storage",
            )

            cooling_action = get_action_value(
                action_names,
                action_array,
                "cooling_device",
            )

            row = {
                "step": logged_step,
                "absolute_step": absolute_step,
                "building_id": building_index + 1,
                "district_target": district_target,
                "battery_action": battery_action,
                "cooling_action": cooling_action,
            }

            row.update(engineered_features)

            building_rows.append(row)
            current_step_building_features.append(engineered_features)

        # ==========================================
        # Portfolio-level row
        # ==========================================
        portfolio_net_load = sum(
            observations[i][load_indices[i]]
            for i in range(number_of_buildings)
        )

        portfolio_solar_generation = sum(
            features["solar_gen"] for features in current_step_building_features
        )

        portfolio_true_load = sum(
            features["true_load"] for features in current_step_building_features
        )

        average_indoor_temp = np.mean(
            [features["indoor_temp"] for features in current_step_building_features]
        )

        average_outdoor_temp = np.mean(
            [features["outdoor_temp"] for features in current_step_building_features]
        )

        average_rolling_temp_3h = np.mean(
            [features["rolling_temp_3h"] for features in current_step_building_features]
        )

        average_cooling_setpoint = np.mean(
            [features["cooling_setpoint"] for features in current_step_building_features]
        )

        average_battery_action = np.mean(
            [
                get_action_value(env.action_names[i], actions[i], "electrical_storage")
                for i in range(number_of_buildings)
            ]
        )

        average_cooling_action = np.mean(
            [
                get_action_value(env.action_names[i], actions[i], "cooling_device")
                for i in range(number_of_buildings)
            ]
        )

        portfolio_rows.append(
            {
                "step": logged_step,
                "absolute_step": absolute_step,
                "district_target": district_target,
                "portfolio_net_load": portfolio_net_load,
                "portfolio_solar_generation": portfolio_solar_generation,
                "portfolio_true_load": portfolio_true_load,
                "tracking_error": portfolio_net_load - district_target,
                "average_indoor_temp": average_indoor_temp,
                "average_outdoor_temp": average_outdoor_temp,
                "average_rolling_temp_3h": average_rolling_temp_3h,
                "average_cooling_setpoint": average_cooling_setpoint,
                "average_battery_action": average_battery_action,
                "average_cooling_action": average_cooling_action,
            }
        )

        done = is_done(terminated, truncated)
        logged_step += 1

    # ==========================================
    # Save outputs
    # ==========================================
    building_df = pd.DataFrame(building_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)

    if len(portfolio_df) == 0:
        raise RuntimeError("No baseline results were collected.")

    actual = portfolio_df["portfolio_net_load"].to_numpy()
    target = portfolio_df["district_target"].to_numpy()

    nmbe, cv_rmse = calculate_metrics(actual, target)

    building_csv_path = OUTPUT_DIR / "tx_real_baseline_building_features.csv"
    portfolio_csv_path = OUTPUT_DIR / "tx_real_baseline_portfolio_features.csv"
    metrics_path = OUTPUT_DIR / "tx_real_baseline_metrics.txt"

    building_df.to_csv(building_csv_path, index=False)
    portfolio_df.to_csv(portfolio_csv_path, index=False)

    with open(metrics_path, "w") as file:
        file.write("Texas Real Baseline Metrics\n")
        file.write("===========================\n")
        file.write(f"Evaluation samples: {len(portfolio_df)} hourly steps\n")
        file.write(f"NMBE: {nmbe:.2f} %\n")
        file.write(f"CV-RMSE: {cv_rmse:.2f} %\n")

    print("\n" + "=" * 70)
    print("Real baseline extraction complete")
    print("=" * 70)
    print(f"Building-level feature rows:  {len(building_df)}")
    print(f"Portfolio-level feature rows: {len(portfolio_df)}")
    print(f"NMBE:     {nmbe:.2f} %")
    print(f"CV-RMSE:  {cv_rmse:.2f} %")
    print(f"Saved building features to:  {building_csv_path}")
    print(f"Saved portfolio features to: {portfolio_csv_path}")
    print(f"Saved metrics to:            {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    extract_real_baseline()