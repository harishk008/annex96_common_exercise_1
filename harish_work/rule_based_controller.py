import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Path setup
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
HARISH_WORK_DIR = REPO_ROOT / "harish_work"
OUTPUT_DIR = HARISH_WORK_DIR / "outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
PLOT_STEPS = 72             # first 3 days × 24 hourly steps/day
TARGET_EPSILON = 1e-6       # threshold for detecting active target period


class RuleBasedTexasController:
    """
    Rule-based controller for the Texas CityLearn dataset.

    Texas active actions:
    - electrical_storage
    - cooling_device

    Controller idea:
    1. During solar hours, charge battery and pre-cool buildings.
    2. During evening peak hours, discharge battery and reduce cooling use.
    3. During normal hours, behave like a simple thermostat.
    """

    def __init__(self, action_space, observation_names, action_names):
        self.action_space = action_space
        self.action_names = action_names

        number_of_buildings = len(action_space)

        if isinstance(observation_names[0], list):
            per_building_observation_names = observation_names[:number_of_buildings]
        else:
            per_building_observation_names = [observation_names] * number_of_buildings

        self.engineers = [
            FeatureEngineer(names) for names in per_building_observation_names
        ]

    def predict(self, observations):
        """
        Create one action vector for each building.
        """
        actions = []

        for building_index, space in enumerate(self.action_space):
            building_obs = observations[building_index]
            features = self.engineers[building_index].process_observation(building_obs)

            building_action = np.zeros(space.shape[0], dtype=float)

            battery_action, cooling_action = self._rule_logic(features)

            self._set_action(
                building_index,
                building_action,
                "electrical_storage",
                battery_action,
            )

            self._set_action(
                building_index,
                building_action,
                "cooling_device",
                cooling_action,
            )

            building_action = np.clip(
                building_action,
                space.low.astype(float),
                space.high.astype(float),
            )

            actions.append(building_action.tolist())

        return actions

    def _rule_logic(self, features):
        """
        Main Texas rule-based control logic.
        """

        hour = features["hour"]
        solar_gen = features["solar_gen"]
        indoor_temp = features["indoor_temp"]
        cooling_setpoint = features["cooling_setpoint"]
        rolling_temp_3h = features["rolling_temp_3h"]

        battery_action = 0.0
        cooling_action = 0.0

        # --------------------------------------------------
        # Strategy A: Solar pre-cooling
        # --------------------------------------------------
        if 10.0 <= hour < 16.0 and solar_gen > 1.0:
            battery_action = 0.10

            if indoor_temp > cooling_setpoint - 2.0:
                cooling_action = 0.50
            else:
                cooling_action = 0.00

        # --------------------------------------------------
        # Strategy B: Evening peak shaving
        # --------------------------------------------------
        elif 18.0 <= hour < 22.0:
            battery_action = -0.30

            if rolling_temp_3h > 30.0:
                temperature_tolerance = 2.0
            else:
                temperature_tolerance = 3.0

            if indoor_temp > cooling_setpoint + temperature_tolerance:
                cooling_action = 0.40
            else:
                cooling_action = 0.00

        # --------------------------------------------------
        # Strategy C: Normal thermostat behavior
        # --------------------------------------------------
        else:
            battery_action = 0.00

            if indoor_temp > cooling_setpoint:
                cooling_action = 0.30
            else:
                cooling_action = 0.00

        return battery_action, cooling_action

    def _set_action(self, building_index, action_array, action_name, value):
        """
        Set action by name if that action exists for the building.
        """
        names = self.action_names[building_index]

        if action_name in names:
            action_index = names.index(action_name)
            action_array[action_index] = value


def get_load_index(observation_names):
    """
    Find net electricity consumption index for each building.
    """
    return observation_names.index("net_electricity_consumption")


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
    directly near the active target period.

    Important:
    The temporary schema is saved inside harish_work/outputs,
    but the actual dataset CSV files are still inside DATASET_DIR.
    Therefore, root_directory must be set to the absolute dataset path.
    """

    env_start_step = max(target_start_step - 1, 0)
    target_end_step = target_start_step + evaluation_steps - 1

    with open(SCHEMA_PATH, "r") as file:
        schema = json.load(file)

    schema["simulation_start_time_step"] = int(env_start_step)
    schema["simulation_end_time_step"] = int(target_end_step)

    # Critical fix:
    # Without this, CityLearn looks for building CSV files inside harish_work/outputs.
    schema["root_directory"] = str(DATASET_DIR)

    active_schema_path = OUTPUT_DIR / "tx_active_target_schema.json"

    with open(active_schema_path, "w") as file:
        json.dump(schema, file, indent=4)

    return active_schema_path, env_start_step, target_end_step


def calculate_metrics(actual, target):
    """
    Calculate NMBE and CV-RMSE.

    Only active target points are used because the target can be zero
    outside the active tracking period.
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


def is_done(terminated, truncated):
    """
    Safely handle bool or list-like termination flags.
    """
    return bool(np.any(terminated)) or bool(np.any(truncated))


def run_simulation():
    print("=" * 70)
    print("Running Texas Rule-Based Controller")
    print("=" * 70)

    print(f"Original schema path: {SCHEMA_PATH}")
    print(f"District target path: {DISTRICT_TARGET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Load target profile first
    # ==========================================
    district_target_series = load_district_target()
    print(f"\nLoaded district target profile with {len(district_target_series)} time steps.")

    target_start_step = find_first_active_target_step(district_target_series)
    target_end_step = target_start_step + EVALUATION_STEPS - 1

    print(f"First active target step: {target_start_step}")
    print(f"Evaluation target window: step {target_start_step} to step {target_end_step}")

    if target_end_step >= len(district_target_series):
        raise ValueError(
            "Evaluation window is longer than available district target profile."
        )

    # ==========================================
    # Create temporary schema for active target period
    # ==========================================
    active_schema_path, env_start_step, env_end_step = create_active_period_schema(
        target_start_step=target_start_step,
        evaluation_steps=EVALUATION_STEPS,
    )

    print(f"Temporary active schema: {active_schema_path}")
    print(f"Environment window: step {env_start_step} to step {env_end_step}")

    # ==========================================
    # Create CityLearn environment
    # ==========================================
    env = CityLearnEnv(schema=str(active_schema_path))
    observations, _ = env.reset()

    number_of_buildings = len(env.action_space)

    print(f"\nNumber of buildings: {number_of_buildings}")
    print(f"Action names for Building 1: {env.action_names[0]}")
    print(f"Number of observations for Building 1: {len(env.observation_names[0])}")

    # ==========================================
    # Create controller
    # ==========================================
    controller = RuleBasedTexasController(
        action_space=env.action_space,
        observation_names=env.observation_names,
        action_names=env.action_names,
    )

    load_indices = [
        get_load_index(env.observation_names[i])
        for i in range(number_of_buildings)
    ]

    # ==========================================
    # Storage for results
    # ==========================================
    history = {
        "step": [],
        "absolute_step": [],
        "portfolio_load": [],
        "district_target": [],
    }

    done = False
    logged_step = 0

    print(f"\nRunning evaluation for {EVALUATION_STEPS} hourly steps...")

    # ==========================================
    # Evaluation period
    # ==========================================
    while not done and logged_step < EVALUATION_STEPS:
        actions = controller.predict(observations)
        observations, rewards, terminated, truncated, info = env.step(actions)

        absolute_step = target_start_step + logged_step

        portfolio_load = sum(
            observations[i][load_indices[i]]
            for i in range(number_of_buildings)
        )

        district_target = district_target_series[absolute_step]

        history["step"].append(logged_step)
        history["absolute_step"].append(absolute_step)
        history["portfolio_load"].append(portfolio_load)
        history["district_target"].append(district_target)

        done = is_done(terminated, truncated)
        logged_step += 1

    # ==========================================
    # Convert results to dataframe
    # ==========================================
    results = pd.DataFrame(history)

    if len(results) == 0:
        raise RuntimeError("No evaluation results were collected.")

    actual = results["portfolio_load"].to_numpy()
    target = results["district_target"].to_numpy()
    target_name = "district_target.csv"

    # ==========================================
    # Calculate metrics
    # ==========================================
    nmbe, cv_rmse = calculate_metrics(actual, target)

    print("\n" + "=" * 70)
    print("Rule-Based Texas Controller Results")
    print("=" * 70)
    print(f"Evaluation target: {target_name}")
    print(f"Evaluation samples: {len(results)} hourly steps")
    print(f"NMBE:     {nmbe:.2f} %")
    print(f"CV-RMSE:  {cv_rmse:.2f} %")
    print("=" * 70)

    # ==========================================
    # Save CSV
    # ==========================================
    csv_path = OUTPUT_DIR / "tx_rule_based_controller_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV to: {csv_path}")

    # ==========================================
    # Save plot
    # ==========================================
    plot_steps = min(PLOT_STEPS, len(results))

    plt.figure(figsize=(14, 6))

    plt.plot(
        results["step"].iloc[:plot_steps],
        actual[:plot_steps],
        label="Actual portfolio load",
        linewidth=2,
    )

    plt.plot(
        results["step"].iloc[:plot_steps],
        target[:plot_steps],
        label=f"Target load ({target_name})",
        linestyle="--",
        linewidth=2,
    )

    plt.title("Texas Rule-Based Controller: First 3 Days of Active Target Period")
    plt.xlabel("Time Step (1-hour intervals)")
    plt.ylabel("Net Grid Load (kW)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "tx_rule_based_controller_graph.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    run_simulation()