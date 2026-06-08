from pathlib import Path
import sys
import json
import tempfile
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from citylearn.citylearn import CityLearnEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "annex96_ce1_tx_neighborhood"
SCHEMA_FILE = DATASET_DIR / "schema.json"
DISTRICT_TARGET_FILE = DATASET_DIR / "district_target.csv"

START_STEP = 3624
TEST_STEPS = 24


def create_temp_schema():
    with open(SCHEMA_FILE, "r") as f:
        schema = json.load(f)

    schema["root_directory"] = str(DATASET_DIR)
    schema["central_agent"] = True
    schema["simulation_start_time_step"] = START_STEP
    schema["simulation_end_time_step"] = START_STEP + TEST_STEPS

    temp_dir = Path(tempfile.mkdtemp(prefix="debug_action_sign_"))
    temp_schema = temp_dir / "schema.json"

    with open(temp_schema, "w") as f:
        json.dump(schema, f, indent=4)

    return temp_schema


def get_live_portfolio_load(env):
    value = env.net_electricity_consumption

    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[-1])

    return float(value)


def make_action(env, storage_value=0.0, cooling_value=0.0):
    action_names = env.action_names[0]
    lows = env.action_space[0].low
    highs = env.action_space[0].high

    action = []

    for i, name in enumerate(action_names):
        if "electrical_storage" in name.lower():
            value = storage_value
        elif "cooling_device" in name.lower():
            value = cooling_value
        else:
            value = 0.0

        value = float(np.clip(value, lows[i], highs[i]))
        action.append(value)

    return [action]


def run_case(case_name, storage_value):
    env = CityLearnEnv(schema=str(create_temp_schema()))
    env.reset()

    loads = []

    for _ in range(TEST_STEPS):
        action = make_action(env, storage_value=storage_value, cooling_value=0.0)
        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        loads.append(get_live_portfolio_load(env))

        if done:
            break

    print(f"\n=== {case_name} ===")
    print(f"Storage action: {storage_value}")
    print(f"Mean load: {np.mean(loads):.4f}")
    print(f"Min load:  {np.min(loads):.4f}")
    print(f"Max load:  {np.max(loads):.4f}")
    print(f"First 5 loads: {[round(x, 3) for x in loads[:5]]}")


def main():
    print("Testing battery action sign over first 24 active hours...")

    run_case("Zero battery", storage_value=0.0)
    run_case("Positive battery action", storage_value=0.5)
    run_case("Negative battery action", storage_value=-0.5)


if __name__ == "__main__":
    main()