import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


# ==========================================
# Path setup
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
HARISH_WORK_DIR = REPO_ROOT / "harish_work"
OUTPUT_DIR = HARISH_WORK_DIR / "outputs"

INPUT_BUILDING_CSV = OUTPUT_DIR / "tx_real_baseline_building_features.csv"
INPUT_PORTFOLIO_CSV = OUTPUT_DIR / "tx_real_baseline_portfolio_features.csv"

ML_OUTPUT_DIR = OUTPUT_DIR / "ml_datasets"

BUILDING_ML_CSV = ML_OUTPUT_DIR / "tx_building_ml_features.csv"
PORTFOLIO_ML_CSV = ML_OUTPUT_DIR / "tx_portfolio_ml_features.csv"

METADATA_JSON = ML_OUTPUT_DIR / "tx_feature_metadata.json"
README_TXT = ML_OUTPUT_DIR / "README_feature_generation.txt"


# ==========================================
# Configuration
# ==========================================
RANDOM_STATE = 42
PCA_COMPONENTS = 3
ICA_COMPONENTS = 3
ENTROPY_WINDOW = 24       # 24 hourly samples = 1 day
ENTROPY_BINS = 10
EPSILON = 1e-9


# ==========================================
# Utility functions
# ==========================================
def safe_divide(numerator, denominator):
    """
    Safe division to avoid division by zero.
    """
    return numerator / (np.abs(denominator) + EPSILON)


def shannon_entropy(values, bins=10):
    """
    Calculate Shannon entropy for a numeric window.

    The signal is discretized into bins first, then entropy is calculated.
    Higher entropy means the signal is more variable / less predictable.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) <= 1:
        return 0.0

    if np.allclose(values, values[0]):
        return 0.0

    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def add_group_lag_features(df, group_col, signal_cols, lags):
    """
    Add lag features for each building separately.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df.groupby(group_col)[col].shift(lag)

            # Fill first lagged values with current building's first valid values.
            df[lag_col] = df.groupby(group_col)[lag_col].bfill()
            df[lag_col] = df[lag_col].fillna(df[col])

    return df


def add_group_rolling_features(df, group_col, signal_cols, windows):
    """
    Add rolling mean and rolling standard deviation for each building.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        for window in windows:
            mean_col = f"{col}_rolling_mean_{window}h"
            std_col = f"{col}_rolling_std_{window}h"

            df[mean_col] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

            df[std_col] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.rolling(window=window, min_periods=2).std())
            )

            df[std_col] = df[std_col].fillna(0.0)

    return df


def add_group_entropy_features(df, group_col, signal_cols, window=24, bins=10):
    """
    Add rolling entropy features for each building.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        entropy_col = f"{col}_entropy_{window}h"

        df[entropy_col] = (
            df.groupby(group_col)[col]
            .transform(
                lambda x: x.rolling(window=window, min_periods=2)
                .apply(lambda y: shannon_entropy(y, bins=bins), raw=False)
            )
        )

        df[entropy_col] = df[entropy_col].fillna(0.0)

    return df


def add_portfolio_lag_features(df, signal_cols, lags):
    """
    Add lag features for portfolio-level time series.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df[col].shift(lag)
            df[lag_col] = df[lag_col].bfill()
            df[lag_col] = df[lag_col].fillna(df[col])

    return df


def add_portfolio_rolling_features(df, signal_cols, windows):
    """
    Add rolling mean and rolling standard deviation for portfolio-level signals.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        for window in windows:
            mean_col = f"{col}_rolling_mean_{window}h"
            std_col = f"{col}_rolling_std_{window}h"

            df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
            df[std_col] = df[col].rolling(window=window, min_periods=2).std()
            df[std_col] = df[std_col].fillna(0.0)

    return df


def add_portfolio_entropy_features(df, signal_cols, window=24, bins=10):
    """
    Add rolling entropy features for portfolio-level signals.
    """
    df = df.copy()

    for col in signal_cols:
        if col not in df.columns:
            continue

        entropy_col = f"{col}_entropy_{window}h"

        df[entropy_col] = (
            df[col]
            .rolling(window=window, min_periods=2)
            .apply(lambda y: shannon_entropy(y, bins=bins), raw=False)
        )

        df[entropy_col] = df[entropy_col].fillna(0.0)

    return df


def add_pca_ica_features(df, feature_cols, prefix, pca_components=3, ica_components=3):
    """
    Add PCA and ICA components to a dataframe.

    PCA and ICA are fitted on standardized numeric features.
    Original columns are preserved.
    """
    df = df.copy()

    existing_feature_cols = [col for col in feature_cols if col in df.columns]

    if len(existing_feature_cols) == 0:
        print(f"Warning: no valid feature columns found for {prefix} PCA/ICA.")
        return df, [], []

    numeric_data = df[existing_feature_cols].copy()
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.fillna(numeric_data.median(numeric_only=True))
    numeric_data = numeric_data.fillna(0.0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    max_components = min(
        pca_components,
        ica_components,
        scaled_data.shape[1],
        scaled_data.shape[0] - 1,
    )

    if max_components < 1:
        print(f"Warning: not enough data for {prefix} PCA/ICA.")
        return df, [], []

    # -----------------------------
    # PCA
    # -----------------------------
    pca = PCA(n_components=max_components, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(scaled_data)

    pca_cols = []
    for i in range(max_components):
        col_name = f"{prefix}_pca_{i + 1}"
        df[col_name] = pca_values[:, i]
        pca_cols.append(col_name)

    # -----------------------------
    # ICA
    # -----------------------------
    ica_cols = []

    try:
        ica = FastICA(
            n_components=max_components,
            random_state=RANDOM_STATE,
            max_iter=1000,
            whiten="unit-variance",
        )

        ica_values = ica.fit_transform(scaled_data)

        for i in range(max_components):
            col_name = f"{prefix}_ica_{i + 1}"
            df[col_name] = ica_values[:, i]
            ica_cols.append(col_name)

    except Exception as error:
        print(f"Warning: ICA failed for {prefix}. Reason: {error}")

        for i in range(max_components):
            col_name = f"{prefix}_ica_{i + 1}"
            df[col_name] = 0.0
            ica_cols.append(col_name)

    explained_variance = pca.explained_variance_ratio_.tolist()
    print(f"{prefix.upper()} PCA explained variance ratio: {explained_variance}")

    return df, pca_cols, ica_cols


# ==========================================
# Building-level ML feature generation
# ==========================================
def generate_building_ml_features(building_df):
    """
    Generate ML-ready building-level features.

    Input shape:
    one row = one building at one hourly time step.
    """
    df = building_df.copy()

    # Sort for correct time-series features.
    df = df.sort_values(["building_id", "step"]).reset_index(drop=True)

    number_of_buildings = df["building_id"].nunique()

    # -----------------------------
    # Basic engineered features
    # -----------------------------
    df["per_building_target"] = df["district_target"] / number_of_buildings

    df["tracking_error_proxy"] = df["net_load"] - df["per_building_target"]

    df["thermal_delta"] = df["indoor_temp"] - df["cooling_setpoint"]

    df["solar_to_true_load_ratio"] = safe_divide(
        df["solar_gen"],
        df["true_load"],
    )

    df["net_to_true_load_ratio"] = safe_divide(
        df["net_load"],
        df["true_load"],
    )

    df["is_solar_hour"] = ((df["hour"] >= 10) & (df["hour"] < 16)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    # -----------------------------
    # Lag features
    # -----------------------------
    lag_signals = [
        "net_load",
        "true_load",
        "solar_gen",
        "indoor_temp",
        "outdoor_temp",
        "tracking_error_proxy",
        "battery_action",
        "cooling_action",
    ]

    df = add_group_lag_features(
        df=df,
        group_col="building_id",
        signal_cols=lag_signals,
        lags=[1, 2, 3],
    )

    # -----------------------------
    # Ramp features
    # -----------------------------
    df["net_load_ramp"] = df["net_load"] - df["net_load_lag_1"]
    df["true_load_ramp"] = df["true_load"] - df["true_load_lag_1"]
    df["solar_ramp"] = df["solar_gen"] - df["solar_gen_lag_1"]
    df["tracking_error_proxy_ramp"] = (
        df["tracking_error_proxy"] - df["tracking_error_proxy_lag_1"]
    )

    # -----------------------------
    # Rolling features
    # -----------------------------
    rolling_signals = [
        "net_load",
        "true_load",
        "solar_gen",
        "indoor_temp",
        "tracking_error_proxy",
    ]

    df = add_group_rolling_features(
        df=df,
        group_col="building_id",
        signal_cols=rolling_signals,
        windows=[3, 6, 24],
    )

    # -----------------------------
    # Entropy features
    # -----------------------------
    entropy_signals = [
        "net_load",
        "true_load",
        "solar_gen",
        "tracking_error_proxy",
    ]

    df = add_group_entropy_features(
        df=df,
        group_col="building_id",
        signal_cols=entropy_signals,
        window=ENTROPY_WINDOW,
        bins=ENTROPY_BINS,
    )

    # -----------------------------
    # PCA / ICA features
    # -----------------------------
    pca_ica_input_cols = [
        "sin_hour",
        "cos_hour",
        "outdoor_temp",
        "rolling_temp_3h",
        "net_load",
        "solar_gen",
        "true_load",
        "indoor_temp",
        "cooling_setpoint",
        "battery_action",
        "cooling_action",
        "tracking_error_proxy",
        "thermal_delta",
        "solar_to_true_load_ratio",
        "net_to_true_load_ratio",
        "net_load_lag_1",
        "net_load_lag_2",
        "net_load_lag_3",
        "true_load_lag_1",
        "solar_gen_lag_1",
        "indoor_temp_lag_1",
        "tracking_error_proxy_lag_1",
        "net_load_ramp",
        "true_load_ramp",
        "solar_ramp",
        "tracking_error_proxy_ramp",
        "net_load_rolling_mean_3h",
        "net_load_rolling_std_3h",
        "true_load_rolling_mean_3h",
        "solar_gen_rolling_mean_3h",
        "indoor_temp_rolling_mean_3h",
        "tracking_error_proxy_rolling_mean_3h",
        "tracking_error_proxy_rolling_std_3h",
    ]

    df, pca_cols, ica_cols = add_pca_ica_features(
        df=df,
        feature_cols=pca_ica_input_cols,
        prefix="building",
        pca_components=PCA_COMPONENTS,
        ica_components=ICA_COMPONENTS,
    )

    return df, pca_cols, ica_cols


# ==========================================
# Portfolio-level ML feature generation
# ==========================================
def generate_portfolio_ml_features(portfolio_df):
    """
    Generate ML-ready portfolio-level features.

    Input shape:
    one row = one hourly portfolio state.
    """
    df = portfolio_df.copy()

    df = df.sort_values("step").reset_index(drop=True)

    # -----------------------------
    # Basic engineered features
    # -----------------------------
    df["load_to_target_ratio"] = safe_divide(
        df["portfolio_net_load"],
        df["district_target"],
    )

    df["true_load_to_target_ratio"] = safe_divide(
        df["portfolio_true_load"],
        df["district_target"],
    )

    df["solar_to_true_load_ratio"] = safe_divide(
        df["portfolio_solar_generation"],
        df["portfolio_true_load"],
    )

    df["absolute_tracking_error"] = np.abs(df["tracking_error"])

    df["squared_tracking_error"] = df["tracking_error"] ** 2

    # Reconstruct hour from absolute step for time flags.
    df["hour_of_day"] = df["absolute_step"] % 24

    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    df["is_solar_hour"] = (
        (df["hour_of_day"] >= 10) & (df["hour_of_day"] < 16)
    ).astype(int)

    df["is_evening_peak"] = (
        (df["hour_of_day"] >= 18) & (df["hour_of_day"] < 22)
    ).astype(int)

    df["is_night"] = (
        (df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)
    ).astype(int)

    # -----------------------------
    # Lag features
    # -----------------------------
    lag_signals = [
        "portfolio_net_load",
        "portfolio_true_load",
        "portfolio_solar_generation",
        "district_target",
        "tracking_error",
        "average_indoor_temp",
        "average_outdoor_temp",
        "average_battery_action",
        "average_cooling_action",
    ]

    df = add_portfolio_lag_features(
        df=df,
        signal_cols=lag_signals,
        lags=[1, 2, 3],
    )

    # -----------------------------
    # Ramp features
    # -----------------------------
    df["portfolio_net_load_ramp"] = (
        df["portfolio_net_load"] - df["portfolio_net_load_lag_1"]
    )

    df["district_target_ramp"] = (
        df["district_target"] - df["district_target_lag_1"]
    )

    df["tracking_error_ramp"] = (
        df["tracking_error"] - df["tracking_error_lag_1"]
    )

    df["portfolio_solar_generation_ramp"] = (
        df["portfolio_solar_generation"] - df["portfolio_solar_generation_lag_1"]
    )

    # -----------------------------
    # Rolling features
    # -----------------------------
    rolling_signals = [
        "portfolio_net_load",
        "portfolio_true_load",
        "portfolio_solar_generation",
        "district_target",
        "tracking_error",
        "average_indoor_temp",
        "average_outdoor_temp",
    ]

    df = add_portfolio_rolling_features(
        df=df,
        signal_cols=rolling_signals,
        windows=[3, 6, 24],
    )

    # -----------------------------
    # Entropy features
    # -----------------------------
    entropy_signals = [
        "portfolio_net_load",
        "portfolio_true_load",
        "portfolio_solar_generation",
        "district_target",
        "tracking_error",
    ]

    df = add_portfolio_entropy_features(
        df=df,
        signal_cols=entropy_signals,
        window=ENTROPY_WINDOW,
        bins=ENTROPY_BINS,
    )

    # -----------------------------
    # PCA / ICA features
    # -----------------------------
    pca_ica_input_cols = [
        "sin_hour",
        "cos_hour",
        "portfolio_net_load",
        "portfolio_true_load",
        "portfolio_solar_generation",
        "district_target",
        "tracking_error",
        "average_indoor_temp",
        "average_outdoor_temp",
        "average_rolling_temp_3h",
        "average_cooling_setpoint",
        "average_battery_action",
        "average_cooling_action",
        "load_to_target_ratio",
        "true_load_to_target_ratio",
        "solar_to_true_load_ratio",
        "absolute_tracking_error",
        "portfolio_net_load_lag_1",
        "portfolio_net_load_lag_2",
        "portfolio_net_load_lag_3",
        "district_target_lag_1",
        "tracking_error_lag_1",
        "portfolio_net_load_ramp",
        "district_target_ramp",
        "tracking_error_ramp",
        "portfolio_net_load_rolling_mean_3h",
        "portfolio_net_load_rolling_std_3h",
        "tracking_error_rolling_mean_3h",
        "tracking_error_rolling_std_3h",
        "district_target_rolling_mean_3h",
    ]

    df, pca_cols, ica_cols = add_pca_ica_features(
        df=df,
        feature_cols=pca_ica_input_cols,
        prefix="portfolio",
        pca_components=PCA_COMPONENTS,
        ica_components=ICA_COMPONENTS,
    )

    return df, pca_cols, ica_cols


# ==========================================
# README and metadata generation
# ==========================================
def write_metadata(building_df, portfolio_df, building_pca_cols, building_ica_cols,
                   portfolio_pca_cols, portfolio_ica_cols):
    """
    Save a JSON metadata file for the generated datasets.
    """
    metadata = {
        "project": "Texas CityLearn ML Feature Dataset",
        "input_files": {
            "building_level": str(INPUT_BUILDING_CSV),
            "portfolio_level": str(INPUT_PORTFOLIO_CSV),
        },
        "output_files": {
            "building_ml_features": str(BUILDING_ML_CSV),
            "portfolio_ml_features": str(PORTFOLIO_ML_CSV),
            "readme": str(README_TXT),
        },
        "dataset_description": {
            "building_level_rows": int(len(building_df)),
            "portfolio_level_rows": int(len(portfolio_df)),
            "building_level_columns": int(len(building_df.columns)),
            "portfolio_level_columns": int(len(portfolio_df.columns)),
            "time_step": "1 hour",
            "evaluation_window": "First active district target period",
            "number_of_buildings": int(building_df["building_id"].nunique())
            if "building_id" in building_df.columns
            else None,
        },
        "feature_groups": {
            "physical_features": [
                "net_load",
                "solar_gen",
                "true_load",
                "indoor_temp",
                "outdoor_temp",
                "cooling_setpoint",
            ],
            "controller_features": [
                "battery_action",
                "cooling_action",
                "average_battery_action",
                "average_cooling_action",
            ],
            "time_features": [
                "hour",
                "sin_hour",
                "cos_hour",
                "hour_of_day",
                "is_solar_hour",
                "is_evening_peak",
                "is_night",
            ],
            "lag_features": "Features ending in _lag_1, _lag_2, _lag_3",
            "ramp_features": "Features ending in _ramp",
            "rolling_features": "Features containing rolling_mean or rolling_std",
            "entropy_features": "Features containing entropy_24h",
            "pca_features": building_pca_cols + portfolio_pca_cols,
            "ica_features": building_ica_cols + portfolio_ica_cols,
        },
        "notes": [
            "Original physical features are preserved.",
            "PCA and ICA were computed on standardized selected numeric features.",
            "Entropy was computed using a 24-hour rolling window.",
            "Building-level tracking_error_proxy uses district_target divided equally across 25 buildings.",
            "Portfolio-level tracking_error uses portfolio_net_load - district_target.",
        ],
    }

    with open(METADATA_JSON, "w") as file:
        json.dump(metadata, file, indent=4)


def write_readme():
    """
    Write a README text file explaining the feature generation process.
    """
    readme_text = f"""
Texas CityLearn ML Feature Dataset
==================================

Purpose
-------
This folder contains ML-ready feature datasets generated from the cleaned Texas
CityLearn / Annex 96 baseline simulation.

The goal is to provide enriched datasets that can be used by the team to build:
- reinforcement learning models,
- supervised machine learning models,
- forecasting models,
- feature-importance analysis,
- target-tracking analysis.

The generated datasets preserve the original physical features and add new
time-series, statistical, PCA, ICA, and entropy-based features.


Input Files
-----------
The feature generation script reads:

1. tx_real_baseline_building_features.csv
   - Building-level baseline dataset.
   - One row represents one building at one hourly time step.
   - Expected size: 25 buildings x 168 hours = 4200 rows.

2. tx_real_baseline_portfolio_features.csv
   - Portfolio-level baseline dataset.
   - One row represents the full portfolio at one hourly time step.
   - Expected size: 168 rows.


Output Files
------------
The generated ML-ready files are:

1. tx_building_ml_features.csv
   - Building-level ML dataset.
   - Useful for building-level ML models and local controller learning.
   - Each row represents one building at one hourly time step.

2. tx_portfolio_ml_features.csv
   - Portfolio-level ML dataset.
   - Useful for portfolio-level target tracking, forecasting, and RL training.
   - Each row represents one hourly portfolio state.

3. tx_feature_metadata.json
   - Machine-readable description of the generated datasets.

4. README_feature_generation.txt
   - This explanation file.


Feature Groups
--------------
The generated datasets include the following groups of features.

1. Original Physical Features
-----------------------------
These are the main physical signals from the CityLearn simulation:

- net_load
- solar_gen
- true_load
- indoor_temp
- outdoor_temp
- cooling_setpoint
- portfolio_net_load
- portfolio_solar_generation
- portfolio_true_load
- district_target

These are kept because they are directly meaningful for energy control.

2. Controller Action Features
-----------------------------
The baseline controller actions are included:

- battery_action
- cooling_action
- average_battery_action
- average_cooling_action

These allow the ML/RL model to understand what action was applied by the
rule-based baseline controller.

3. Time Features
----------------
The following time-related features are included:

- hour
- hour_of_day
- sin_hour
- cos_hour
- is_solar_hour
- is_evening_peak
- is_night

The sine and cosine hour features encode the daily cycle more correctly than
plain hour values. For example, hour 23 and hour 0 are close in real time, and
sin/cos encoding helps a model understand this cyclic relationship.

4. Tracking Error Features
--------------------------
Portfolio-level tracking error is:

    tracking_error = portfolio_net_load - district_target

Building-level tracking error proxy is:

    tracking_error_proxy = net_load - district_target / number_of_buildings

The building-level version is only a proxy because the district target is defined
for the full portfolio, not for individual buildings.

5. Lag Features
---------------
Lag features store previous signal values:

- _lag_1
- _lag_2
- _lag_3

For hourly data, these correspond to the previous 1, 2, and 3 hours.

Examples:

- net_load_lag_1
- solar_gen_lag_1
- tracking_error_lag_1
- portfolio_net_load_lag_1
- district_target_lag_1

These features help ML models understand short-term memory and trends.

6. Ramp Features
----------------
Ramp features measure how quickly a signal changes:

    ramp = current value - previous value

Examples:

- net_load_ramp
- true_load_ramp
- solar_ramp
- portfolio_net_load_ramp
- district_target_ramp
- tracking_error_ramp

These are useful because a controller must respond not only to the current load,
but also to how quickly the load or target is changing.

7. Rolling Window Features
--------------------------
Rolling features summarize recent history over short time windows:

- 3-hour rolling mean
- 6-hour rolling mean
- 24-hour rolling mean
- rolling standard deviation

Examples:

- net_load_rolling_mean_3h
- net_load_rolling_std_3h
- tracking_error_rolling_mean_3h
- tracking_error_rolling_std_3h

Rolling means show recent average behavior.
Rolling standard deviations show short-term variability.

8. Entropy Features
-------------------
Entropy features measure how variable or unpredictable a signal is over a
24-hour rolling window.

Examples:

- net_load_entropy_24h
- solar_gen_entropy_24h
- tracking_error_entropy_24h
- portfolio_net_load_entropy_24h

Higher entropy means the signal changes in a less predictable way. This may be
useful for RL models because periods with high entropy may require more adaptive
control.

9. PCA Features
---------------
PCA stands for Principal Component Analysis.

PCA creates compressed components from correlated numerical features. The PCA
features are:

- building_pca_1
- building_pca_2
- building_pca_3
- portfolio_pca_1
- portfolio_pca_2
- portfolio_pca_3

These components can reduce redundancy between features such as temperature,
cooling action, load, solar generation, and tracking error.

10. ICA Features
----------------
ICA stands for Independent Component Analysis.

ICA attempts to separate hidden independent patterns from the feature set. The
ICA features are:

- building_ica_1
- building_ica_2
- building_ica_3
- portfolio_ica_1
- portfolio_ica_2
- portfolio_ica_3

These may help separate hidden effects such as solar-driven behavior,
temperature-driven behavior, load-driven behavior, and control-action behavior.


How to Use These Datasets
-------------------------

For portfolio-level RL or ML:
Use:

    tx_portfolio_ml_features.csv

Recommended target/output variables:
- tracking_error
- portfolio_net_load
- district_target
- average_battery_action
- average_cooling_action

For building-level ML:
Use:

    tx_building_ml_features.csv

Useful for:
- building behavior modeling,
- local load prediction,
- clustering buildings,
- studying action-response behavior.

Important Notes
---------------
1. The dataset uses hourly time steps.
2. The selected evaluation window is the first active district target period.
3. The Texas dataset uses the following active actions:
   - electrical_storage
   - cooling_device
4. The Vermont dataset is not used in this feature generation process.
5. Original physical features are not removed.
6. PCA and ICA features are additional features, not replacements.
7. Entropy is calculated using a 24-hour rolling window.
8. The building-level tracking error is a proxy because the target is defined at
   the portfolio level.


Suggested Team Usage
--------------------
The team can test different feature sets:

1. Physical features only
2. Physical + lag/ramp features
3. Physical + rolling/entropy features
4. Physical + PCA/ICA features
5. All features together

This allows comparison of whether PCA, ICA, and entropy improve ML/RL model
performance.


Generated Files Location
------------------------
All generated files are saved in:

    {ML_OUTPUT_DIR}

"""

    with open(README_TXT, "w") as file:
        file.write(readme_text.strip())


# ==========================================
# Main pipeline
# ==========================================
def main():
    print("=" * 70)
    print("Generating Texas ML Feature Datasets")
    print("=" * 70)

    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_BUILDING_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_BUILDING_CSV}")

    if not INPUT_PORTFOLIO_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PORTFOLIO_CSV}")

    print(f"Reading building-level input:  {INPUT_BUILDING_CSV}")
    print(f"Reading portfolio-level input: {INPUT_PORTFOLIO_CSV}")

    building_df = pd.read_csv(INPUT_BUILDING_CSV)
    portfolio_df = pd.read_csv(INPUT_PORTFOLIO_CSV)

    print(f"\nInput building rows:  {len(building_df)}")
    print(f"Input portfolio rows: {len(portfolio_df)}")

    # Generate ML features.
    building_ml_df, building_pca_cols, building_ica_cols = generate_building_ml_features(
        building_df
    )

    portfolio_ml_df, portfolio_pca_cols, portfolio_ica_cols = generate_portfolio_ml_features(
        portfolio_df
    )

    # Save datasets.
    building_ml_df.to_csv(BUILDING_ML_CSV, index=False)
    portfolio_ml_df.to_csv(PORTFOLIO_ML_CSV, index=False)

    # Save metadata and README.
    write_metadata(
        building_df=building_ml_df,
        portfolio_df=portfolio_ml_df,
        building_pca_cols=building_pca_cols,
        building_ica_cols=building_ica_cols,
        portfolio_pca_cols=portfolio_pca_cols,
        portfolio_ica_cols=portfolio_ica_cols,
    )

    write_readme()

    print("\n" + "=" * 70)
    print("ML feature generation complete")
    print("=" * 70)
    print(f"Building ML rows:     {len(building_ml_df)}")
    print(f"Building ML columns:  {len(building_ml_df.columns)}")
    print(f"Portfolio ML rows:    {len(portfolio_ml_df)}")
    print(f"Portfolio ML columns: {len(portfolio_ml_df.columns)}")
    print()
    print(f"Saved building ML dataset to:  {BUILDING_ML_CSV}")
    print(f"Saved portfolio ML dataset to: {PORTFOLIO_ML_CSV}")
    print(f"Saved metadata to:             {METADATA_JSON}")
    print(f"Saved README to:               {README_TXT}")
    print("=" * 70)


if __name__ == "__main__":
    main()