"""
Group-aware ML controller for Annex 96 / CityLearn Texas neighborhood.

Goal
----
Use behavior-based building clusters to apply different actions to groups of
buildings while still tracking the district target at portfolio level.

Expected inputs
---------------
1. Trained response model:
   harish_work/outputs/ml_controller/tx_response_model_hist_gradient_boosting.joblib

2. Action-response dataset used for model feature fallback:
   harish_work/outputs/ml_controller/tx_action_response_dataset.csv

3. Building cluster assignments from cluster_buildings.py:
   harish_work/outputs/ml_controller/building_clusters/building_cluster_assignments.csv

The cluster CSV must contain at least:
   building_id, controller_cluster

Example run
-----------
python harish_work/group_aware_ml_controller.py \
  --schema data/datasets/annex96_ce1_tx_neighborhood/schema.json \
  --district-target data/datasets/annex96_ce1_tx_neighborhood/district_target.csv \
  --clusters harish_work/outputs/ml_controller/building_clusters/building_cluster_assignments.csv \
  --model harish_work/outputs/ml_controller/tx_response_model_hist_gradient_boosting.joblib \
  --response-dataset harish_work/outputs/ml_controller/tx_action_response_dataset.csv \
  --start-step 3624 \
  --horizon 2160 \
  --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:  # keep script runnable without wandb installed/login
    wandb = None

try:
    from citylearn.citylearn import CityLearnEnv
except ImportError as exc:
    raise ImportError(
        "Could not import CityLearnEnv. Run this from the project root with the "
        "CityLearn environment activated."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_BATTERY_CANDIDATES = [-0.30, -0.15, 0.00, 0.15, 0.30]
DEFAULT_COOLING_CANDIDATES = [0.00, 0.25, 0.50, 0.75, 1.00]

# Candidate cluster offsets around the central action.
# These are intentionally small so the controller does not create a wild
# closed-loop distribution shift immediately.
BATTERY_CLUSTER_OFFSETS = [-0.15, 0.00, 0.15]
COOLING_CLUSTER_OFFSETS = [-0.25, 0.00, 0.25]

BATTERY_MIN = -0.30
BATTERY_MAX = 0.30
COOLING_MIN = 0.00
COOLING_MAX = 1.00

EPS = 1e-9


@dataclass
class ControllerResult:
    nmbE_percent: float
    cv_rmse_percent: float
    mae: float
    rmse: float
    max_abs_error: float
    output_csv: Path


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def read_district_target(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Accept several likely column names. The existing project used
    # district_target in earlier outputs, but some files may use target names.
    possible_target_cols = [
        "district_target",
        "district_load_target",
        "target",
        "target_load",
        "reference_load",
        "district_target_load",
    ]
    target_col = next((c for c in possible_target_cols if c in df.columns), None)
    if target_col is None:
        raise ValueError(
            f"Could not find target column in {path}. Columns are: {list(df.columns)}"
        )

    if target_col != "district_target":
        df = df.rename(columns={target_col: "district_target"})

    return df


def get_target_at_step(target_df: pd.DataFrame, step: int) -> float:
    """Return district target for an absolute CityLearn step."""
    if "time_step" in target_df.columns:
        row = target_df.loc[target_df["time_step"] == step]
        if not row.empty:
            return safe_float(row.iloc[0]["district_target"])

    # Fallback: treat row index as absolute step.
    if step < len(target_df):
        return safe_float(target_df.iloc[step]["district_target"])

    # Final fallback: if evaluation starts at active step and CSV was already cut.
    local_step = step - int(target_df.index.min())
    if 0 <= local_step < len(target_df):
        return safe_float(target_df.iloc[local_step]["district_target"])

    raise IndexError(f"Step {step} is outside district target file length {len(target_df)}")


def load_clusters(path: Path, expected_n_buildings: int = 25) -> pd.DataFrame:
    clusters = pd.read_csv(path)

    if "building_id" not in clusters.columns:
        # Common fallback if index was saved unnamed.
        unnamed = [c for c in clusters.columns if c.lower().startswith("unnamed")]
        if unnamed:
            clusters = clusters.rename(columns={unnamed[0]: "building_id"})
        else:
            raise ValueError(
                f"Cluster file must contain building_id. Columns: {list(clusters.columns)}"
            )

    if "controller_cluster" not in clusters.columns:
        possible = ["cluster", "cluster_id", "kmeans_cluster", "building_cluster"]
        found = next((c for c in possible if c in clusters.columns), None)
        if found is None:
            raise ValueError(
                "Cluster file must contain controller_cluster or a recognizable "
                f"cluster column. Columns: {list(clusters.columns)}"
            )
        clusters = clusters.rename(columns={found: "controller_cluster"})

    clusters = clusters[["building_id", "controller_cluster"]].copy()
    clusters["building_id"] = clusters["building_id"].astype(int)
    clusters["controller_cluster"] = clusters["controller_cluster"].astype(int)
    clusters = clusters.sort_values("building_id").reset_index(drop=True)

    if len(clusters) != expected_n_buildings:
        warnings.warn(
            f"Expected {expected_n_buildings} buildings in cluster file, found {len(clusters)}."
        )

    # Your cluster CSV uses building_id = 1..25, while Python/CityLearn
    # observation indices are 0..24. Detect and store the offset.
    min_id = int(clusters["building_id"].min())
    max_id = int(clusters["building_id"].max())
    if min_id == 1 and max_id == expected_n_buildings:
        id_base = 1
    elif min_id == 0 and max_id == expected_n_buildings - 1:
        id_base = 0
    else:
        id_base = min_id
        warnings.warn(
            f"Unusual building_id range {min_id}..{max_id}. Assuming observation index + {id_base}."
        )

    clusters.attrs["id_base"] = id_base
    print("=== Loaded Building Clusters ===")
    print(clusters)
    print("cluster counts:")
    print(clusters["controller_cluster"].value_counts().sort_index())
    print(f"Detected building_id base: {id_base}")

    return clusters


def maybe_start_wandb(enabled: bool, entity: str, project: str, config: Dict[str, Any]):
    if not enabled:
        return None
    if wandb is None:
        warnings.warn("wandb is not installed. Continuing without W&B logging.")
        return None
    return wandb.init(
    entity=entity,
    project=project,
    name="Our-ML-PI-Cluster-Controller",
    config=config,
    tags=["final", "ml-pi", "group-aware", "texas"],
)

# -----------------------------------------------------------------------------
# Model handling
# -----------------------------------------------------------------------------

def unwrap_model(model_object: Any) -> Tuple[Any, Optional[List[str]], Dict[str, Any]]:
    """
    Support either a plain sklearn estimator or a saved dict bundle.
    """
    metadata: Dict[str, Any] = {}
    feature_columns: Optional[List[str]] = None

    if isinstance(model_object, dict):
        metadata = {k: v for k, v in model_object.items() if k != "model"}
        model = model_object.get("model") or model_object.get("estimator")
        if model is None:
            raise ValueError("Model bundle dict does not contain key 'model' or 'estimator'.")
        for key in ["feature_columns", "features", "input_columns"]:
            if key in model_object:
                feature_columns = list(model_object[key])
                break
    else:
        model = model_object

    if feature_columns is None and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    return model, feature_columns, metadata


def infer_feature_columns(
    model: Any,
    explicit_feature_columns: Optional[List[str]],
    response_dataset_path: Path,
) -> List[str]:
    if explicit_feature_columns:
        return explicit_feature_columns

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    response_df = pd.read_csv(response_dataset_path, nrows=10)
    drop_cols = {
        "next_portfolio_load",
        "target_next_portfolio_load",
        "actual_next_portfolio_load",
        "actual_portfolio_load",
        "predicted_next_load",
        "error",
        "absolute_error",
    }
    numeric_cols = [
        c for c in response_df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(response_df[c])
    ]
    if not numeric_cols:
        raise ValueError("Could not infer numeric feature columns from response dataset.")
    return numeric_cols


def build_model_input(
    feature_columns: Sequence[str],
    step: int,
    hour: int,
    day: int,
    month: int,
    current_portfolio_load: float,
    district_target: float,
    battery_action: float,
    cooling_action: float,
    cluster_summary: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build one model row and fill only the columns known by the trained model.

    This is intentionally defensive: different training scripts may have used
    slightly different names. Unknown model columns are filled with 0.0.
    """
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    day_sin = math.sin(2.0 * math.pi * day / 365.0)
    day_cos = math.cos(2.0 * math.pi * day / 365.0)

    values: Dict[str, float] = {
        "time_step": step,
        "step": step,
        "hour": hour,
        "day": day,
        "month": month,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "portfolio_load": current_portfolio_load,
        "current_portfolio_load": current_portfolio_load,
        "actual_portfolio_load": current_portfolio_load,
        "net_electricity_consumption": current_portfolio_load,
        "district_target": district_target,
        "target": district_target,
        "reference_load": district_target,
        "battery_action": battery_action,
        "electrical_storage_action": battery_action,
        "cooling_action": cooling_action,
        "cooling_device_action": cooling_action,
        "mean_battery_action": battery_action,
        "mean_cooling_action": cooling_action,
        "portfolio_battery_action": battery_action,
        "portfolio_cooling_action": cooling_action,
        "target_error": district_target - current_portfolio_load,
        "load_target_error": current_portfolio_load - district_target,
        "abs_target_error": abs(district_target - current_portfolio_load),
    }

    if cluster_summary:
        values.update(cluster_summary)

    row = {col: safe_float(values.get(col, 0.0)) for col in feature_columns}
    return pd.DataFrame([row], columns=list(feature_columns))


# -----------------------------------------------------------------------------
# CityLearn observation/action helpers
# -----------------------------------------------------------------------------

def get_nested_names(env: CityLearnEnv, attr_name: str) -> Optional[List[List[str]]]:
    """Read env.observation_names/env.action_names defensively."""
    names = getattr(env, attr_name, None)
    if names is None:
        return None
    try:
        # CityLearn usually returns list[list[str]], one list per building.
        if len(names) > 0 and isinstance(names[0], (list, tuple)):
            return [list(x) for x in names]
        return [list(names) for _ in range(len(env.buildings))]
    except Exception:
        return None


def value_from_obs(
    obs_i: Sequence[float],
    obs_names_i: Optional[Sequence[str]],
    possible_names: Sequence[str],
    default: float = 0.0,
) -> float:
    if obs_names_i is None:
        return default
    for name in possible_names:
        if name in obs_names_i:
            idx = obs_names_i.index(name)
            if idx < len(obs_i):
                return safe_float(obs_i[idx], default)
    return default


def extract_live_building_state(
    observations: Sequence[Sequence[float]],
    observation_names: Optional[List[List[str]]],
) -> pd.DataFrame:
    rows = []
    for b_idx, obs_i in enumerate(observations):
        names_i = observation_names[b_idx] if observation_names and b_idx < len(observation_names) else None
        rows.append({
            "building_id": b_idx,
            "soc": value_from_obs(
                obs_i, names_i,
                ["electrical_storage_soc", "electric_storage_soc", "storage_soc", "battery_soc"],
                default=0.5,
            ),
            "indoor_temp": value_from_obs(
                obs_i, names_i,
                ["indoor_dry_bulb_temperature", "indoor_temperature", "zone_temp"],
                default=22.0,
            ),
            "cooling_setpoint": value_from_obs(
                obs_i, names_i,
                [
                    "indoor_dry_bulb_temperature_cooling_set_point",
                    "cooling_set_point",
                    "cooling_setpoint",
                ],
                default=24.0,
            ),
            "heating_setpoint": value_from_obs(
                obs_i, names_i,
                [
                    "indoor_dry_bulb_temperature_heating_set_point",
                    "heating_set_point",
                    "heating_setpoint",
                ],
                default=20.0,
            ),
            "net_load": value_from_obs(
                obs_i, names_i,
                [
                    "net_electricity_consumption",
                    "net_electricity_consumption_without_storage",
                    "electrical_demand",
                    "non_shiftable_load",
                ],
                default=0.0,
            ),
            "cooling_demand": value_from_obs(
                obs_i, names_i,
                ["cooling_demand", "cooling_load"],
                default=0.0,
            ),
            "solar_generation": value_from_obs(
                obs_i, names_i,
                ["solar_generation", "solar_power", "pv_generation"],
                default=0.0,
            ),
        })
    return pd.DataFrame(rows)


def portfolio_load_from_observations(
    observations: Sequence[Sequence[float]],
    observation_names: Optional[List[List[str]]],
) -> float:
    state = extract_live_building_state(observations, observation_names)
    return float(state["net_load"].sum())


def get_action_index(action_names_i: Optional[Sequence[str]], possible_names: Sequence[str]) -> Optional[int]:
    if action_names_i is None:
        return None
    for name in possible_names:
        if name in action_names_i:
            return action_names_i.index(name)
    return None


def build_citylearn_actions(
    env: CityLearnEnv,
    action_names: Optional[List[List[str]]],
    battery_actions: Sequence[float],
    cooling_actions: Sequence[float],
) -> List[List[float]]:
    """Return CityLearn action list, preserving the action order per building."""
    actions: List[List[float]] = []
    n_buildings = len(battery_actions)

    for b_idx in range(n_buildings):
        if action_names is not None and b_idx < len(action_names):
            names_i = action_names[b_idx]
            action_i = [0.0 for _ in names_i]

            battery_idx = get_action_index(
                names_i,
                ["electrical_storage", "electric_storage", "battery", "electrical_storage_action"],
            )
            cooling_idx = get_action_index(
                names_i,
                ["cooling_device", "cooling", "cooling_device_action"],
            )

            if battery_idx is not None:
                action_i[battery_idx] = safe_float(battery_actions[b_idx])
            if cooling_idx is not None:
                action_i[cooling_idx] = safe_float(cooling_actions[b_idx])

            actions.append(action_i)
        else:
            # Texas setup discussed earlier uses two actions:
            # [electrical_storage, cooling_device]
            actions.append([safe_float(battery_actions[b_idx]), safe_float(cooling_actions[b_idx])])

    return actions


# -----------------------------------------------------------------------------
# Controller logic
# -----------------------------------------------------------------------------
class GroupAwareMLController:
    def __init__(
        self,
        model: Any,
        feature_columns: Sequence[str],
        clusters: pd.DataFrame,
        battery_candidates: Sequence[float] = DEFAULT_BATTERY_CANDIDATES,
        cooling_candidates: Sequence[float] = DEFAULT_COOLING_CANDIDATES,
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.clusters = clusters.copy()
        self.id_base = int(clusters.attrs.get("id_base", 0))

        self.battery_candidates = list(battery_candidates)
        self.cooling_candidates = list(cooling_candidates)

        self.building_to_cluster = dict(
            zip(
                self.clusters["building_id"].astype(int),
                self.clusters["controller_cluster"].astype(int),
            )
        )

        self.cluster_ids = sorted(
            self.clusters["controller_cluster"].unique().astype(int).tolist()
        )

        # Online state for tracking.
        self.integral_error = 0.0
        self.previous_battery_action = 0.0
        self.previous_cooling_action = 0.0

        # Online prediction-bias correction for logging/model proposal.
        self.prediction_bias = 0.0
        self.last_predicted_next_load: Optional[float] = None

    def update_prediction_bias(self, current_portfolio_load: float) -> None:
        """
        Update online ML prediction bias using previous prediction and current observed load.
        """
        if self.last_predicted_next_load is None:
            return

        prediction_error = current_portfolio_load - self.last_predicted_next_load
        self.prediction_bias = 0.90 * self.prediction_bias + 0.10 * prediction_error
        self.prediction_bias = clip(self.prediction_bias, -30.0, 30.0)

    def choose_ml_proposal(
        self,
        step: int,
        hour: int,
        day: int,
        month: int,
        current_portfolio_load: float,
        district_target: float,
    ) -> Tuple[float, float, float]:
        """
        ML proposal only.

        The trained response model is useful, but closed-loop tests showed
        distribution shift. Therefore this output is not blindly applied.
        It is blended with a robust PI tracking controller.
        """
        best = None
        live_error = district_target - current_portfolio_load
        abs_live_error = abs(live_error)

        for battery_action in self.battery_candidates:
            for cooling_action in self.cooling_candidates:
                x = build_model_input(
                    self.feature_columns,
                    step=step,
                    hour=hour,
                    day=day,
                    month=month,
                    current_portfolio_load=current_portfolio_load,
                    district_target=district_target,
                    battery_action=battery_action,
                    cooling_action=cooling_action,
                )

                raw_pred = safe_float(self.model.predict(x)[0])
                corrected_pred = raw_pred + self.prediction_bias

                score = abs(corrected_pred - district_target)

                # Penalize aggressive cooling unless the error is large.
                if cooling_action >= 0.50 and abs_live_error < 30.0:
                    score += 5.0

                # Penalize extreme battery action near the target.
                if abs(battery_action) >= 0.30 and abs_live_error < 15.0:
                    score += 3.0

                candidate = (
                    score,
                    battery_action,
                    cooling_action,
                    corrected_pred,
                )

                if best is None or candidate[0] < best[0]:
                    best = candidate

        assert best is not None
        _, battery_action, cooling_action, corrected_pred = best

        return float(battery_action), float(cooling_action), float(corrected_pred)

    def summarize_clusters(self, live_state: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize live CityLearn observations per controller cluster.
        """
        df = live_state.copy()

        df["controller_cluster"] = (
            df["building_id"].map(self.building_to_cluster).fillna(0).astype(int)
        )

        summary = (
            df.groupby("controller_cluster")
            .agg(
                n_buildings=("building_id", "count"),
                mean_soc=("soc", "mean"),
                mean_indoor_temp=("indoor_temp", "mean"),
                mean_cooling_setpoint=("cooling_setpoint", "mean"),
                mean_net_load=("net_load", "mean"),
                sum_net_load=("net_load", "sum"),
                mean_cooling_demand=("cooling_demand", "mean"),
            )
            .reset_index()
        )

        summary["thermal_headroom"] = (
            summary["mean_cooling_setpoint"] - summary["mean_indoor_temp"]
        )

        return summary

    def tracking_base_action(
        self,
        current_portfolio_load: float,
        district_target: float,
        ml_battery: float,
        ml_cooling: float,
        cluster_summary: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Stable portfolio-level PI tracking controller.

        Sign convention from your action-response dataset:
        - Higher battery action increases portfolio load.
        - Higher cooling action increases portfolio load.

        Therefore:
        - If target > actual, increase battery/cooling.
        - If target < actual, reduce battery/cooling or discharge battery.
        """
        error = district_target - current_portfolio_load
        abs_error = abs(error)

        # Leaky integral prevents long-term bias while avoiding wind-up.
        self.integral_error = 0.97 * self.integral_error + error
        self.integral_error = clip(self.integral_error, -120.0, 120.0)

        if cluster_summary.empty:
            mean_indoor_temp = 22.0
            mean_cooling_sp = 24.0
        else:
            mean_indoor_temp = safe_float(cluster_summary["mean_indoor_temp"].mean(), 22.0)
            mean_cooling_sp = safe_float(cluster_summary["mean_cooling_setpoint"].mean(), 24.0)

        # PI proposal. These gains are intentionally moderate.
        battery_pi = (
            self.previous_battery_action
            + 0.0060 * error
            + 0.0008 * self.integral_error
        )

        cooling_pi = self.previous_cooling_action

        if error > 0.0:
            # Need more load.
            # Use cooling only if it is thermally reasonable.
            if abs_error > 35.0 and mean_indoor_temp >= mean_cooling_sp - 2.0:
                cooling_pi += 0.15
            elif abs_error > 20.0 and mean_indoor_temp >= mean_cooling_sp - 1.0:
                cooling_pi += 0.08
            else:
                cooling_pi *= 0.95
        else:
            # Need less load.
            cooling_pi *= 0.65

        # Blend ML proposal and PI feedback.
        # PI gets higher weight because closed-loop ML alone had strong bias.
        battery_desired = 0.25 * ml_battery + 0.75 * battery_pi
        cooling_desired = 0.25 * ml_cooling + 0.75 * cooling_pi

        # Deadband near target: avoid oscillation.
        if abs_error <= 4.0:
            battery_desired = 0.70 * self.previous_battery_action + 0.30 * ml_battery
            cooling_desired = 0.70 * self.previous_cooling_action + 0.30 * min(ml_cooling, 0.25)
            self.integral_error *= 0.90

        # Smoothing.
        battery = 0.55 * self.previous_battery_action + 0.45 * battery_desired
        cooling = 0.60 * self.previous_cooling_action + 0.40 * cooling_desired

        # Safety limits.
        battery = clip(battery, BATTERY_MIN, BATTERY_MAX)
        cooling = clip(cooling, COOLING_MIN, 0.50)

        self.previous_battery_action = battery
        self.previous_cooling_action = cooling

        return battery, cooling

    def zero_mean_offsets(self, cluster_summary: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cluster offsets that redistribute action without changing the
        portfolio-level mean too much.
        """
        summary = cluster_summary.copy()
        total_buildings = float(summary["n_buildings"].sum())

        if total_buildings <= 0:
            summary["battery_offset"] = 0.0
            summary["cooling_offset"] = 0.0
            return summary

        weighted_mean_soc = (
            summary["mean_soc"] * summary["n_buildings"]
        ).sum() / total_buildings

        temp_error = summary["mean_indoor_temp"] - summary["mean_cooling_setpoint"]
        weighted_temp_error = (
            temp_error * summary["n_buildings"]
        ).sum() / total_buildings

        # Same formula works for charging/discharging:
        # lower SOC gets relatively higher battery action,
        # higher SOC gets relatively lower battery action.
        summary["battery_offset"] = -0.08 * (summary["mean_soc"] - weighted_mean_soc)

        # Warmer clusters get more cooling, cooler clusters get less.
        summary["cooling_offset"] = 0.08 * (
            np.clip(temp_error - weighted_temp_error, -2.0, 2.0) / 2.0
        )

        summary["battery_offset"] = summary["battery_offset"].clip(-0.08, 0.08)
        summary["cooling_offset"] = summary["cooling_offset"].clip(-0.08, 0.08)

        return summary

    def recenter_weighted_actions(
        self,
        summary: pd.DataFrame,
        target_battery: float,
        target_cooling: float,
    ) -> pd.DataFrame:
        """
        Recenter cluster actions so their building-weighted mean remains close
        to the portfolio-level action.
        """
        total_buildings = float(summary["n_buildings"].sum())

        if total_buildings <= 0:
            return summary

        current_battery_mean = (
            summary["battery_action"] * summary["n_buildings"]
        ).sum() / total_buildings

        current_cooling_mean = (
            summary["cooling_action"] * summary["n_buildings"]
        ).sum() / total_buildings

        battery_delta = target_battery - current_battery_mean
        cooling_delta = target_cooling - current_cooling_mean

        summary["battery_action"] = (
            summary["battery_action"] + battery_delta
        ).clip(BATTERY_MIN, BATTERY_MAX)

        summary["cooling_action"] = (
            summary["cooling_action"] + cooling_delta
        ).clip(COOLING_MIN, 0.50)

        return summary

    def cluster_adjustments(
        self,
        base_battery: float,
        base_cooling: float,
        cluster_summary: pd.DataFrame,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Apply group-aware redistribution around the base portfolio action.
        """
        result: Dict[int, Tuple[float, float]] = {}

        if cluster_summary.empty:
            return {
                cid: (base_battery, base_cooling)
                for cid in self.cluster_ids
            }

        summary = self.zero_mean_offsets(cluster_summary)

        summary["battery_action"] = base_battery + summary["battery_offset"]
        summary["cooling_action"] = base_cooling + summary["cooling_offset"]

        # Per-cluster safety.
        for idx, row in summary.iterrows():
            mean_soc = safe_float(row.get("mean_soc", 0.5), 0.5)
            indoor_temp = safe_float(row.get("mean_indoor_temp", 22.0), 22.0)
            cooling_sp = safe_float(row.get("mean_cooling_setpoint", 24.0), 24.0)

            battery = safe_float(row["battery_action"])
            cooling = safe_float(row["cooling_action"])

            if mean_soc > 0.95 and battery > 0.0:
                battery = 0.0

            if mean_soc < 0.10 and battery < 0.0:
                battery = 0.0

            if indoor_temp < cooling_sp - 2.0:
                cooling = 0.0

            summary.loc[idx, "battery_action"] = clip(battery, BATTERY_MIN, BATTERY_MAX)
            summary.loc[idx, "cooling_action"] = clip(cooling, COOLING_MIN, 0.50)

        summary = self.recenter_weighted_actions(
            summary=summary,
            target_battery=base_battery,
            target_cooling=base_cooling,
        )

        for _, row in summary.iterrows():
            cid = int(row["controller_cluster"])

            result[cid] = (
                clip(safe_float(row["battery_action"]), BATTERY_MIN, BATTERY_MAX),
                clip(safe_float(row["cooling_action"]), COOLING_MIN, 0.50),
            )

        return result

    def actions_for_buildings(
        self,
        n_buildings: int,
        cluster_action_map: Dict[int, Tuple[float, float]],
    ) -> Tuple[List[float], List[float]]:
        battery_actions = []
        cooling_actions = []

        for b_idx in range(n_buildings):
            building_id = b_idx + self.id_base
            cid = self.building_to_cluster.get(building_id, 0)

            battery, cooling = cluster_action_map.get(cid, (0.0, 0.0))

            battery_actions.append(float(battery))
            cooling_actions.append(float(cooling))

        return battery_actions, cooling_actions

    def predict_for_final_action(
        self,
        step: int,
        hour: int,
        day: int,
        month: int,
        current_portfolio_load: float,
        district_target: float,
        battery_action: float,
        cooling_action: float,
    ) -> float:
        x = build_model_input(
            self.feature_columns,
            step=step,
            hour=hour,
            day=day,
            month=month,
            current_portfolio_load=current_portfolio_load,
            district_target=district_target,
            battery_action=battery_action,
            cooling_action=cooling_action,
        )

        pred = safe_float(self.model.predict(x)[0]) + self.prediction_bias
        self.last_predicted_next_load = pred
        return float(pred)

    def act(
        self,
        observations: Sequence[Sequence[float]],
        observation_names: Optional[List[List[str]]],
        step: int,
        district_target: float,
    ) -> Dict[str, Any]:
        current_portfolio_load = portfolio_load_from_observations(
            observations,
            observation_names,
        )

        self.update_prediction_bias(current_portfolio_load)

        hour = step % 24
        day = step // 24
        month = int((day % 365) // 30 + 1)

        ml_battery, ml_cooling, _ = self.choose_ml_proposal(
            step=step,
            hour=hour,
            day=day,
            month=month,
            current_portfolio_load=current_portfolio_load,
            district_target=district_target,
        )

        live_state = extract_live_building_state(observations, observation_names)
        live_state["building_id"] = live_state["building_id"] + self.id_base

        cluster_summary = self.summarize_clusters(live_state)

        base_battery, base_cooling = self.tracking_base_action(
            current_portfolio_load=current_portfolio_load,
            district_target=district_target,
            ml_battery=ml_battery,
            ml_cooling=ml_cooling,
            cluster_summary=cluster_summary,
        )

        predicted_next_load = self.predict_for_final_action(
            step=step,
            hour=hour,
            day=day,
            month=month,
            current_portfolio_load=current_portfolio_load,
            district_target=district_target,
            battery_action=base_battery,
            cooling_action=base_cooling,
        )

        cluster_action_map = self.cluster_adjustments(
            base_battery=base_battery,
            base_cooling=base_cooling,
            cluster_summary=cluster_summary,
        )

        battery_actions, cooling_actions = self.actions_for_buildings(
            n_buildings=len(observations),
            cluster_action_map=cluster_action_map,
        )

        return {
            "current_portfolio_load": current_portfolio_load,
            "central_battery_action": base_battery,
            "central_cooling_action": base_cooling,
            "ml_battery_action": ml_battery,
            "ml_cooling_action": ml_cooling,
            "predicted_next_load": predicted_next_load,
            "prediction_bias": self.prediction_bias,
            "integral_error": self.integral_error,
            "cluster_action_map": cluster_action_map,
            "cluster_summary": cluster_summary,
            "battery_actions": battery_actions,
            "cooling_actions": cooling_actions,
        }
# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
def unpack_reset_result(reset_result: Any) -> Any:
    """Support both old and new reset return formats."""
    if isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict):
        return reset_result[0]
    return reset_result

def read_schema_time_bounds(schema_path: Path) -> Tuple[int, Optional[int]]:
    """Read simulation start/end step from CityLearn schema if available."""
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    start_step = int(schema.get("simulation_start_time_step", 0))

    end_step = schema.get("simulation_end_time_step", None)
    if end_step is not None:
        end_step = int(end_step)

    return start_step, end_step

def reset_to_start_step(
    env: CityLearnEnv,
    start_step: int,
    schema_start_step: int = 0,
) -> Sequence[Sequence[float]]:
    """
    Reset env and advance only relative to the schema start step.

    Example:
    - If schema starts at 0 and start_step=3624, advance 3624 steps.
    - If schema already starts at 3624 and start_step=3624, advance 0 steps.
    """
    reset_result = env.reset()
    observations = unpack_reset_result(reset_result)

    action_names = get_nested_names(env, "action_names")
    n_buildings = len(env.buildings)

    zero_battery = [0.0] * n_buildings
    zero_cooling = [0.0] * n_buildings
    zero_actions = build_citylearn_actions(env, action_names, zero_battery, zero_cooling)

    advance_steps = start_step - schema_start_step

    if advance_steps < 0:
        warnings.warn(
            f"Requested start_step={start_step}, but schema starts at "
            f"schema_start_step={schema_start_step}. Using reset state directly."
        )
        advance_steps = 0

    print(f"Reset observation length: {len(observations)}")
    print(f"Number of buildings from env: {n_buildings}")
    print(f"Zero action outer length: {len(zero_actions)}")
    print(f"Schema start step: {schema_start_step}")
    print(f"Requested absolute start step: {start_step}")
    print(f"Advance steps after reset: {advance_steps}")

    for i in range(advance_steps):
        step_result = env.step(zero_actions)
        observations, reward, done, info = unpack_step_result(step_result)

        if done:
            raise RuntimeError(
                f"Environment ended while advancing to start_step={start_step}. "
                f"Stopped after {i + 1}/{advance_steps} advance steps. "
                f"Schema may already be shortened."
            )

    return observations


def unpack_step_result(step_result: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Support both Gym old and Gymnasium style step returns."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, reward, done, info
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, reward, bool(done), info
    raise ValueError(f"Unexpected env.step return length: {len(step_result)}")


def compute_metrics(actual: Sequence[float], target: Sequence[float]) -> Dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    error = actual_arr - target_arr

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    max_abs_error = float(np.max(np.abs(error)))

    target_mean = float(np.mean(target_arr))
    nmbE_percent = float(100.0 * np.sum(error) / (len(error) * (target_mean + EPS)))
    cv_rmse_percent = float(100.0 * rmse / (target_mean + EPS))

    return {
        "NMBE_percent": nmbE_percent,
        "CV_RMSE_percent": cv_rmse_percent,
        "MAE": mae,
        "RMSE": rmse,
        "Max_Abs_Error": max_abs_error,
    }

def evaluate(
    schema_path: Path,
    district_target_path: Path,
    cluster_path: Path,
    model_path: Path,
    response_dataset_path: Path,
    output_dir: Path,
    start_step: int,
    horizon: int,
    use_wandb: bool,
    wandb_entity: str,
    wandb_project: str,
) -> ControllerResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = read_district_target(district_target_path)
    clusters = load_clusters(cluster_path, expected_n_buildings=25)

    model_object = joblib.load(model_path)
    model, explicit_feature_columns, model_metadata = unwrap_model(model_object)
    feature_columns = infer_feature_columns(
        model,
        explicit_feature_columns,
        response_dataset_path,
    )

    schema_start_step, schema_end_step = read_schema_time_bounds(schema_path)

    print(f"Schema simulation_start_time_step: {schema_start_step}")
    print(f"Schema simulation_end_time_step: {schema_end_step}")

    if schema_end_step is not None:
        available_steps = schema_end_step - start_step + 1

        if available_steps <= 0:
            raise ValueError(
                f"Requested start_step={start_step}, but schema ends at "
                f"simulation_end_time_step={schema_end_step}."
            )

        if horizon > available_steps:
            warnings.warn(
                f"Requested horizon={horizon}, but schema only has "
                f"{available_steps} steps from start_step={start_step}. "
                f"Using horizon={available_steps}."
            )
            horizon = available_steps

    env = CityLearnEnv(str(schema_path))
    observation_names = get_nested_names(env, "observation_names")
    action_names = get_nested_names(env, "action_names")

    controller = GroupAwareMLController(
        model=model,
        feature_columns=feature_columns,
        clusters=clusters,
    )

    run = maybe_start_wandb(
        enabled=use_wandb,
        entity=wandb_entity,
        project=wandb_project,
        config={
            "controller": "group_aware_ml_controller",
            "schema": str(schema_path),
            "district_target": str(district_target_path),
            "clusters": str(cluster_path),
            "model": str(model_path),
            "response_dataset": str(response_dataset_path),
            "start_step": start_step,
            "horizon": horizon,
            "schema_start_step": schema_start_step,
            "schema_end_step": schema_end_step,
            "n_clusters": int(clusters["controller_cluster"].nunique()),
            "feature_columns": feature_columns,
            "model_metadata": model_metadata,
        },
    )

    observations = reset_to_start_step(
        env=env,
        start_step=start_step,
        schema_start_step=schema_start_step,
    )

    rows: List[Dict[str, Any]] = []
    actual_loads: List[float] = []
    targets: List[float] = []
    comfort_violation_count = 0
    comfort_total_count = 0

    for local_t in range(horizon):
        absolute_step = start_step + local_t
        target = get_target_at_step(target_df, absolute_step)

        decision = controller.act(
            observations=observations,
            observation_names=observation_names,
            step=absolute_step,
            district_target=target,
        )

        actions = build_citylearn_actions(
            env=env,
            action_names=action_names,
            battery_actions=decision["battery_actions"],
            cooling_actions=decision["cooling_actions"],
        )

        if local_t == 0:
            print(f"Live observation length: {len(observations)}")
            print(f"Battery action length: {len(decision['battery_actions'])}")
            print(f"Cooling action length: {len(decision['cooling_actions'])}")
            print(f"CityLearn action outer length: {len(actions)}")
            print(f"Expected number of buildings: {len(env.buildings)}")

        next_observations, reward, done, info = unpack_step_result(env.step(actions))

        actual_next_load = portfolio_load_from_observations(
            next_observations,
            observation_names,
        )
        error = actual_next_load - target
        
        next_live_state = extract_live_building_state(next_observations, observation_names)

        too_hot = next_live_state["indoor_temp"] > next_live_state["cooling_setpoint"]
        too_cold = next_live_state["indoor_temp"] < next_live_state["heating_setpoint"]

        step_comfort_violations = int((too_hot | too_cold).sum())
        step_comfort_total = int(len(next_live_state))

        comfort_violation_count += step_comfort_violations
        comfort_total_count += step_comfort_total

        step_comfort_violation_percent = (
            100.0 * step_comfort_violations / step_comfort_total
            if step_comfort_total > 0
            else 0.0
        )

        actual_loads.append(actual_next_load)
        targets.append(target)

        row = {
            "local_step": local_t,
            "absolute_step": absolute_step,
            "hour": absolute_step % 24,
            "district_target": target,
            "current_portfolio_load": decision["current_portfolio_load"],
            "actual_portfolio_load": actual_next_load,
            "predicted_next_load": decision["predicted_next_load"],
            "prediction_bias": decision.get("prediction_bias", 0.0),
            "integral_error": decision.get("integral_error", 0.0),
            "error": error,
            "absolute_error": abs(error),
            "central_battery_action": decision["central_battery_action"],
            "central_cooling_action": decision["central_cooling_action"],
            "ml_battery_action": decision.get("ml_battery_action", 0.0),
            "ml_cooling_action": decision.get("ml_cooling_action", 0.0),
            "mean_battery_action": float(np.mean(decision["battery_actions"])),
            "mean_cooling_action": float(np.mean(decision["cooling_actions"])),
            "reward": safe_float(reward),
            "temp_comfort_violation_percent_step": step_comfort_violation_percent,
}

        for cid, (battery, cooling) in decision["cluster_action_map"].items():
            row[f"cluster_{cid}_battery_action"] = battery
            row[f"cluster_{cid}_cooling_action"] = cooling

        rows.append(row)

        if run is not None:
            wandb.log({
            "step/local": local_t,
            "step/absolute": absolute_step,

            # Main load tracking names
            "TargetLoad": target,
            "Our-ML-PI-Cluster-Controller": actual_next_load,

            # Clear grouped names
            "load/district_load_target": target,
            "load/actual_portfolio_load": actual_next_load,
            "load/current_portfolio_load": decision["current_portfolio_load"],
            "load/predicted_next_load": decision["predicted_next_load"],

            # Errors
            "error/error": error,
            "error/absolute_error": abs(error),

            # Actions
            "action/central_battery": decision["central_battery_action"],
            "action/central_cooling": decision["central_cooling_action"],
            "action/mean_battery": float(np.mean(decision["battery_actions"])),
            "action/mean_cooling": float(np.mean(decision["cooling_actions"])),

            # Optional controller internals
            "controller/prediction_bias": decision.get("prediction_bias", 0.0),
            "controller/integral_error": decision.get("integral_error", 0.0),
        })

        observations = next_observations

        if done:
            warnings.warn(f"Environment ended early at local step {local_t}.")
            break

    metrics = compute_metrics(actual_loads, targets)
    
    metrics["Temp_Comfort_Violation_percent"] = (
    100.0 * comfort_violation_count / comfort_total_count
    if comfort_total_count > 0
    else 0.0
)

    result_df = pd.DataFrame(rows)
    output_csv = output_dir / "tx_group_aware_ml_controller_results.csv"
    result_df.to_csv(output_csv, index=False)

    metrics_path = output_dir / "tx_group_aware_ml_controller_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Group-Aware ML Controller Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nSaved results: {output_csv}")
    print(f"Saved metrics: {metrics_path}")

    if run is not None:
        final_logs = {f"final/{k}": float(v) for k, v in metrics.items()}

        # Extra clean names for W&B dashboards/plots
        final_logs["NMBE [%]"] = float(metrics["NMBE_percent"])
        final_logs["CV-RMSE [%]"] = float(metrics["CV_RMSE_percent"])
        final_logs["MAE"] = float(metrics["MAE"])
        final_logs["RMSE"] = float(metrics["RMSE"])
        final_logs["Max Abs Error"] = float(metrics["Max_Abs_Error"])

    if "Temp_Comfort_Violation_percent" in metrics:
        final_logs["Temp Comfort violation [%]"] = float(
            metrics["Temp_Comfort_Violation_percent"]
        )

    # Log final metrics to W&B
    wandb.log(final_logs)

    # Force values into W&B run summary
    wandb.run.summary["NMBE [%]"] = float(metrics["NMBE_percent"])
    wandb.run.summary["CV-RMSE [%]"] = float(metrics["CV_RMSE_percent"])
    wandb.run.summary["MAE"] = float(metrics["MAE"])
    wandb.run.summary["RMSE"] = float(metrics["RMSE"])
    wandb.run.summary["Max Abs Error"] = float(metrics["Max_Abs_Error"])

    if "Temp_Comfort_Violation_percent" in metrics:
        wandb.run.summary["Temp Comfort violation [%]"] = float(
            metrics["Temp_Comfort_Violation_percent"]
        )

    wandb.finish()

    return ControllerResult(
        nmbE_percent=metrics["NMBE_percent"],
        cv_rmse_percent=metrics["CV_RMSE_percent"],
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        max_abs_error=metrics["Max_Abs_Error"],
        output_csv=output_csv,
    )
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run group-aware ML controller for CityLearn Texas.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/datasets/annex96_ce1_tx_neighborhood/schema.json"),
    )
    parser.add_argument(
        "--district-target",
        type=Path,
        default=Path("data/datasets/annex96_ce1_tx_neighborhood/district_target.csv"),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("harish_work/outputs/ml_controller/building_clusters/building_cluster_assignments.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("harish_work/outputs/ml_controller/tx_response_model_hist_gradient_boosting.joblib"),
    )
    parser.add_argument(
        "--response-dataset",
        type=Path,
        default=Path("harish_work/outputs/ml_controller/tx_action_response_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("harish_work/outputs/ml_controller/group_aware_controller"),
    )
    parser.add_argument("--start-step", type=int, default=3624)
    parser.add_argument("--horizon", type=int, default=2160)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="CityLearn-TeamB")
    parser.add_argument("--wandb-project", type=str, default="CityLearn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path_name in ["schema", "district_target", "clusters", "model", "response_dataset"]:
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing --{path_name.replace('_', '-')}: {path}")

    evaluate(
        schema_path=args.schema,
        district_target_path=args.district_target,
        cluster_path=args.clusters,
        model_path=args.model,
        response_dataset_path=args.response_dataset,
        output_dir=args.output_dir,
        start_step=args.start_step,
        horizon=args.horizon,
        use_wandb=args.wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
