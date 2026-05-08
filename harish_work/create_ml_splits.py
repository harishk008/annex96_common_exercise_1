from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
SPLIT_DIR = ML_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

BUILDING_FILE = ML_DIR / "tx_building_ml_features.csv"
PORTFOLIO_FILE = ML_DIR / "tx_portfolio_ml_features.csv"


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_time_column(df: pd.DataFrame) -> str:
    for col in ["time_step", "step", "hour_index"]:
        if col in df.columns:
            return col

    raise ValueError("No time column found. Expected one of: time_step, step, hour_index")


def create_time_splits(unique_times: list[int]) -> dict[str, list[int]]:
    n = len(unique_times)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    return {
        "train": train_times,
        "validation": val_times,
        "test": test_times,
    }


def assign_split(df: pd.DataFrame, time_col: str, splits: dict[str, list[int]]) -> pd.DataFrame:
    df = df.copy()

    split_map = {}

    for split_name, times in splits.items():
        for t in times:
            split_map[t] = split_name

    df["split"] = df[time_col].map(split_map)

    missing = df["split"].isna().sum()

    if missing > 0:
        raise ValueError(f"{missing} rows could not be assigned to a split.")

    return df


def save_split_files(df: pd.DataFrame, prefix: str) -> dict:
    summary = {}

    full_file = SPLIT_DIR / f"{prefix}_with_split.csv"
    df.to_csv(full_file, index=False)

    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()
        out_file = SPLIT_DIR / f"{prefix}_{split_name}.csv"
        split_df.to_csv(out_file, index=False)

        summary[split_name] = {
            "rows": int(len(split_df)),
            "file": str(out_file.relative_to(PROJECT_ROOT)),
        }

    summary["full_with_split"] = str(full_file.relative_to(PROJECT_ROOT))

    return summary


def main() -> None:
    building_df = pd.read_csv(BUILDING_FILE)
    portfolio_df = pd.read_csv(PORTFOLIO_FILE)

    portfolio_time_col = get_time_column(portfolio_df)
    building_time_col = get_time_column(building_df)

    unique_times = sorted(portfolio_df[portfolio_time_col].unique().tolist())

    splits = create_time_splits(unique_times)

    portfolio_split = assign_split(portfolio_df, portfolio_time_col, splits)

    # Rename split time keys if building uses a different time-column name.
    if building_time_col == portfolio_time_col:
        building_splits = splits
    else:
        building_splits = splits

    building_split = assign_split(building_df, building_time_col, building_splits)

    portfolio_summary = save_split_files(portfolio_split, "tx_portfolio_ml_features")
    building_summary = save_split_files(building_split, "tx_building_ml_features")

    summary = {
        "split_method": "chronological",
        "reason": "Time-series ML/RL data should not be randomly split because future samples would leak into training.",
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "time_column_portfolio": portfolio_time_col,
        "time_column_building": building_time_col,
        "total_unique_time_steps": len(unique_times),
        "time_step_ranges": {
            split_name: {
                "start": int(times[0]) if len(times) > 0 else None,
                "end": int(times[-1]) if len(times) > 0 else None,
                "count": len(times),
            }
            for split_name, times in splits.items()
        },
        "portfolio": portfolio_summary,
        "building": building_summary,
    }

    summary_file = SPLIT_DIR / "tx_ml_split_summary.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== Split summary ===")
    print(json.dumps(summary, indent=4))

    print(f"\nSaved split files to: {SPLIT_DIR}")


if __name__ == "__main__":
    main()