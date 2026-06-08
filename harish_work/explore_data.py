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
    Create a temporary schema that starts the CityLearn simulation
    directly at the first active Texas district target period.

    The temporary schema is saved in harish_work/outputs, but the real
    dataset files remain in DATASET_DIR.

    Note:
    We extend the environment end step by one extra hour because this script
    logs the current observation first and then calls env.step().
    """

    env_start_step = int(target_start_step)

    # Target window is 168 logged samples:
    # target_start_step ... target_start_step + 167
    target_end_step = int(target_start_step + evaluation_steps - 1)

    # Environment must run one step beyond the final logged target step.
    env_end_step = int(target_start_step + evaluation_steps)

    with open(SCHEMA_PATH, "r") as file:
        schema = json.load(file)

    schema["simulation_start_time_step"] = env_start_step
    schema["simulation_end_time_step"] = env_end_step

    # Important:
    # The temporary schema is stored in outputs/, so we must explicitly
    # tell CityLearn where the real CSV files are.
    schema["root_directory"] = str(DATASET_DIR)

    active_schema_path = OUTPUT_DIR / "tx_feature_extraction_schema.json"

    with open(active_schema_path, "w") as file:
        json.dump(schema, file, indent=4)

    return active_schema_path, env_start_step, target_end_step


def make_zero_actions(action_space):
    """
    Create zero actions for all buildings.

    For feature extraction, we are not trying to control the buildings.
    We only advance the environment and collect observations.
    """
    actions = []

    for space in action_space:
        zero_action = np.zeros(space.shape[0], dtype=float)
        zero_action = np.clip(
            zero_action,
            space.low.astype(float),
            space.high.astype(float),
        )
        actions.append(zero_action.tolist())

    return actions


def is_done(terminated, truncated):
    """
    Safely handle bool or list-like termination flags.
    """
    return bool(np.any(terminated)) or bool(np.any(truncated))


def raw_observation_dict(observation, observation_names):
    """
    Convert raw CityLearn observation array into a dictionary.

    Each raw observation column gets the prefix 'raw_'.
    """
    raw_data = {}

    for index, name in enumerate(observation_names):
        raw_data[f"raw_{name}"] = observation[index]

    return raw_data


def collect_features():
    print("=" * 70)
    print("Texas Feature Extraction")
    print("=" * 70)

    print(f"Original schema path: {SCHEMA_PATH}")
    print(f"District target path: {DISTRICT_TARGET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Load target profile
    # ==========================================
    district_target_series = load_district_target()
    print(f"\nLoaded district target profile with {len(district_target_series)} time steps.")

    target_start_step = find_first_active_target_step(district_target_series)
    target_end_step = target_start_step + EVALUATION_STEPS - 1

    print(f"First active target step: {target_start_step}")
    print(f"Feature extraction window: step {target_start_step} to step {target_end_step}")

    if target_end_step >= len(district_target_series):
        raise ValueError(
            "Feature extraction window is longer than available district target profile."
        )

    # ==========================================
    # Create temporary active-period schema
    # ==========================================
    active_schema_path, env_start_step, env_end_step = create_active_period_schema(
        target_start_step=target_start_step,
        evaluation_steps=EVALUATION_STEPS,
    )

    print(f"Temporary feature schema: {active_schema_path}")
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

    # One FeatureEngineer per building.
    engineers = [
        FeatureEngineer(env.observation_names[i])
        for i in range(number_of_buildings)
    ]

    building_rows = []
    portfolio_rows = []

    done = False
    logged_step = 0

    print(f"\nCollecting features for {EVALUATION_STEPS} hourly steps...")

    while not done and logged_step < EVALUATION_STEPS:
        absolute_step = target_start_step + logged_step
        district_target = district_target_series[absolute_step]

        current_step_building_features = []

        # ==========================================
        # Building-level feature extraction
        # ==========================================
        for building_index in range(number_of_buildings):
            building_obs = observations[building_index]
            observation_names = env.observation_names[building_index]

            engineered_features = engineers[building_index].process_observation(
                building_obs
            )

            raw_features = raw_observation_dict(
                building_obs,
                observation_names,
            )

            row = {
                "step": logged_step,
                "absolute_step": absolute_step,
                "building_id": building_index + 1,
                "district_target": district_target,
            }

            row.update(engineered_features)
            row.update(raw_features)

            building_rows.append(row)
            current_step_building_features.append(engineered_features)

        # ==========================================
        # Portfolio-level feature extraction
        # ==========================================
        portfolio_net_load = sum(
            features["net_load"] for features in current_step_building_features
        )

        portfolio_solar_generation = sum(
            features["solar_gen"] for features in current_step_building_features
        )

        portfolio_true_load = sum(
            features["true_load"] for features in current_step_building_features
        )

        average_outdoor_temp = np.mean(
            [features["outdoor_temp"] for features in current_step_building_features]
        )

        average_rolling_temp_3h = np.mean(
            [features["rolling_temp_3h"] for features in current_step_building_features]
        )

        average_indoor_temp = np.mean(
            [features["indoor_temp"] for features in current_step_building_features]
        )

        average_cooling_setpoint = np.mean(
            [features["cooling_setpoint"] for features in current_step_building_features]
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
                "average_outdoor_temp": average_outdoor_temp,
                "average_rolling_temp_3h": average_rolling_temp_3h,
                "average_indoor_temp": average_indoor_temp,
                "average_cooling_setpoint": average_cooling_setpoint,
            }
        )

        # Advance simulation with zero actions.
        actions = make_zero_actions(env.action_space)
        observations, rewards, terminated, truncated, info = env.step(actions)

        done = is_done(terminated, truncated)
        logged_step += 1

    # ==========================================
    # Save results
    # ==========================================
    building_features_df = pd.DataFrame(building_rows)
    portfolio_features_df = pd.DataFrame(portfolio_rows)

    building_csv_path = OUTPUT_DIR / "tx_building_features_active_period.csv"
    portfolio_csv_path = OUTPUT_DIR / "tx_portfolio_features_active_period.csv"

    building_features_df.to_csv(building_csv_path, index=False)
    portfolio_features_df.to_csv(portfolio_csv_path, index=False)

    print("\n" + "=" * 70)
    print("Feature extraction complete")
    print("=" * 70)
    print(f"Building-level feature rows:  {len(building_features_df)}")
    print(f"Portfolio-level feature rows: {len(portfolio_features_df)}")
    print(f"Saved building features to:  {building_csv_path}")
    print(f"Saved portfolio features to: {portfolio_csv_path}")
    print("=" * 70)

    print("\nBuilding-level columns:")
    print(building_features_df.columns.tolist())

    print("\nPortfolio-level columns:")
    print(portfolio_features_df.columns.tolist())


if __name__ == "__main__":
    collect_features()