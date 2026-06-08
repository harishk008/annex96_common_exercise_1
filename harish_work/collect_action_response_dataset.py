from pathlib import Path
import sys
import json
import tempfile
import random

import numpy as np
import pandas as pd

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

ML_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
PORTFOLIO_FEATURE_FILE = ML_DIR / "tx_portfolio_ml_features.csv"
BUILDING_FEATURE_FILE = ML_DIR / "tx_building_ml_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_controller"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "tx_action_response_dataset.csv"


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

START_STEP = 3624
EVALUATION_DAYS = 90
STEPS_PER_DAY = 24
EVALUATION_STEPS = EVALUATION_DAYS * STEPS_PER_DAY

N_EPISODES = 8
RANDOM_SEED = 42

BATTERY_CANDIDATES = [-0.30, -0.15, 0.0, 0.15, 0.30]
COOLING_CANDIDATES = [0.0, 0.25, 0.50, 0.75, 1.0]


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

    temp_dir = Path(tempfile.mkdtemp(prefix="citylearn_action_response_"))
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


def get_target_series():
    df = pd.read_csv(DISTRICT_TARGET_FILE)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric target column found in district_target.csv")

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


# ------------------------------------------------------------
# Main dataset collection
# ------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n=== Collecting CityLearn action-response dataset ===")

    target = get_target_series()
    portfolio_df = pd.read_csv(PORTFOLIO_FEATURE_FILE)
    building_df = pd.read_csv(BUILDING_FEATURE_FILE)

    portfolio_time_col = get_time_column(portfolio_df)
    building_time_col = get_time_column(building_df)

    portfolio_feature_cols = {
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

    building_feature_cols = {
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

    all_records = []

    for episode in range(N_EPISODES):
        print(f"\nEpisode {episode + 1}/{N_EPISODES}")

        env = CityLearnEnv(schema=str(create_temp_schema()))
        env_reset(env)

        print(f"Central agent: {env.central_agent}")
        print(f"Number of actions: {len(env.action_names[0])}")

        for relative_step in range(EVALUATION_STEPS):
            abs_step = START_STEP + relative_step

            current_load = get_live_portfolio_load(env)

            battery_action = random.choice(BATTERY_CANDIDATES)
            cooling_action = random.choice(COOLING_CANDIDATES)

            action = make_uniform_action(
                env,
                battery_value=battery_action,
                cooling_value=cooling_action,
            )

            obs, reward, done, info = env_step(env, action)

            next_load = get_live_portfolio_load(env)

            portfolio_row = get_row_by_step(
                portfolio_df,
                portfolio_time_col,
                relative_step,
            )

            building_rows = get_building_rows_by_step(
                building_df,
                building_time_col,
                relative_step,
            )

            record = {
                "episode": episode,
                "relative_step": relative_step,
                "absolute_step": abs_step,
                "hour": relative_step % 24,
                "day": relative_step // 24,
                "district_target": float(target.iloc[abs_step]) if abs_step < len(target) else float(target.iloc[-1]),
                "current_portfolio_load": current_load,
                "battery_action": battery_action,
                "cooling_action": cooling_action,
                "next_portfolio_load": next_load,
                "load_change": next_load - current_load,
                "tracking_error_next": next_load - (float(target.iloc[abs_step]) if abs_step < len(target) else float(target.iloc[-1])),
            }

            for out_name, col in portfolio_feature_cols.items():
                record[out_name] = safe_get(portfolio_row, col, default=0.0)

            for out_name, col in building_feature_cols.items():
                if col is not None and col in building_rows.columns:
                    record[out_name] = float(building_rows[col].mean())
                else:
                    record[out_name] = 0.0

            all_records.append(record)

            if relative_step % 240 == 0:
                print(
                    f"  step {relative_step:4d} | "
                    f"load {current_load:8.2f} -> {next_load:8.2f} | "
                    f"battery {battery_action:+.2f} | cooling {cooling_action:.2f}"
                )

            if done:
                break

    response_df = pd.DataFrame(all_records)
    response_df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Saved action-response dataset ===")
    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {len(response_df)}")
    print(f"Columns: {len(response_df.columns)}")

    print("\nQuick summary:")
    print(response_df[[
        "current_portfolio_load",
        "battery_action",
        "cooling_action",
        "next_portfolio_load",
        "load_change",
        "tracking_error_next",
    ]].describe())

    print("\nMean next load by battery action:")
    print(response_df.groupby("battery_action")["next_portfolio_load"].mean())

    print("\nMean next load by cooling action:")
    print(response_df.groupby("cooling_action")["next_portfolio_load"].mean())


if __name__ == "__main__":
    main()