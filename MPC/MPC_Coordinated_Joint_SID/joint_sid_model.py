"""
joint_sid_model.py
------------------
Joint learned state-space/NARX system-identification model for CityLearn.

The model predicts the next values of a compact control state per building:
    T_indoor[k+1]
    electrical_storage_soc[k+1]
    net_electricity_consumption[k+1]
    cooling_electricity_consumption[k+1]

It is trained as a residual model:
    y = Ridge(X) + MLP_residual(X)

Artifacts expected in saved_files/joint_sid/:
    joint_sid_bundle.joblib
    joint_residual_mlp_seed_*.pt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import joblib
import numpy as np
import torch
import torch.nn as nn


TARGET_NAMES = [
    "indoor_dry_bulb_temperature_next",
    "electrical_storage_soc_next",
    "net_electricity_consumption_next",
    "cooling_electricity_consumption_next",
]

STATE_NAMES = [
    "indoor_dry_bulb_temperature",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "cooling_electricity_consumption",
]

ACTION_NAMES_DEFAULT = ["electrical_storage", "cooling_device"]


class JointResidualMLP(nn.Module):
    def __init__(self, n_features: int, n_outputs: int = 4, hidden: int = 384, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, n_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JointSIDModel:
    """Loader/inference wrapper for the joint SID model."""

    def __init__(self, sid_dir: str | Path, device: str | None = None):
        self.sid_dir = Path(sid_dir)
        bundle_path = self.sid_dir / "joint_sid_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing SID bundle: {bundle_path}. Run train_joint_sid.ipynb first.")

        bundle = joblib.load(bundle_path)
        self.feature_names: List[str] = list(bundle["feature_names"])
        self.target_names: List[str] = list(bundle.get("target_names", TARGET_NAMES))
        self.state_names: List[str] = list(bundle.get("state_names", STATE_NAMES))
        self.lags: Tuple[int, ...] = tuple(bundle.get("lags", (1, 2, 3, 6, 12)))
        self.x_scaler = bundle["x_scaler"]
        self.y_scaler = bundle["y_scaler"]
        self.ridge = bundle["ridge"]
        self.model_config = bundle.get("model_config", {"hidden": 384, "depth": 4, "dropout": 0.05})

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: List[JointResidualMLP] = []
        for p in sorted(self.sid_dir.glob("joint_residual_mlp_seed_*.pt")):
            m = JointResidualMLP(
                n_features=len(self.feature_names),
                n_outputs=len(self.target_names),
                hidden=int(self.model_config.get("hidden", 384)),
                depth=int(self.model_config.get("depth", 4)),
                dropout=float(self.model_config.get("dropout", 0.05)),
            ).to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval()
            self.models.append(m)

    def predict_from_feature_dict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        row = np.array([[float(feature_dict.get(c, 0.0)) for c in self.feature_names]], dtype=np.float32)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.x_scaler.transform(row)
        pred_s = np.asarray(self.ridge.predict(Xs), dtype=np.float32)

        if self.models:
            xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            preds = []
            with torch.no_grad():
                for m in self.models:
                    preds.append(m(xt).detach().cpu().numpy())
            pred_s = pred_s + np.mean(preds, axis=0)

        y = self.y_scaler.inverse_transform(pred_s)[0]
        return {name: float(val) for name, val in zip(self.target_names, y)}


# -----------------------------------------------------------------------------
# CityLearn helper utilities
# -----------------------------------------------------------------------------
def normalize_names(env, kind: str = "observation") -> List[List[str]]:
    n = len(env.buildings)
    raw = env.observation_names if kind == "observation" else env.action_names
    if isinstance(raw, list) and len(raw) >= n and isinstance(raw[0], (list, tuple)):
        return [list(x) for x in raw[:n]]

    spaces = env.observation_space if kind == "observation" else env.action_space
    dims = [len(s.low) for s in spaces[:n]]
    out, start = [], 0
    for d in dims:
        out.append(list(raw[start:start + d]))
        start += d
    return out


def obs_to_dicts(obs, obs_names: List[List[str]]) -> List[Dict[str, float]]:
    return [dict(zip(names, np.asarray(o, dtype=float).tolist())) for o, names in zip(obs, obs_names)]


def safe_get(d: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, default)
        if v is None or not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def get_action_bounds(env, action_names: List[List[str]]) -> List[Dict[str, Tuple[float, float]]]:
    out = []
    for i, names in enumerate(action_names):
        low = np.asarray(env.action_space[i].low, dtype=float)
        high = np.asarray(env.action_space[i].high, dtype=float)
        out.append({n: (float(low[j]), float(high[j])) for j, n in enumerate(names)})
    return out


def build_action_vector(action_names_i: List[str], electrical_storage: float, cooling_device: float) -> List[float]:
    vals = []
    for n in action_names_i:
        if n == "electrical_storage":
            vals.append(float(electrical_storage))
        elif n == "cooling_device":
            vals.append(float(cooling_device))
        else:
            vals.append(0.0)
    return vals


def make_time_features(hour: float, month: float) -> Dict[str, float]:
    hour = float(hour)
    month = float(month)
    return {
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
    }


def build_feature_dict_from_history(
    building_id: int,
    current_obs: Dict[str, float],
    current_action: Dict[str, float],
    history: Dict[str, List[float]],
    lags: Sequence[int],
    exog_override: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Build the feature dictionary used by the joint SID model.

    history should contain lists for STATE_NAMES and actions, where the last value is current k.
    """
    f: Dict[str, float] = {}
    f["building_id"] = float(building_id)

    month = safe_get(current_obs, "month", 1.0)
    hour = safe_get(current_obs, "hour", 0.0)
    f.update(make_time_features(hour, month))

    # Current exogenous/buffered observations.
    exog_keys = [
        "outdoor_dry_bulb_temperature",
        "direct_solar_irradiance",
        "solar_generation",
        "non_shiftable_load",
        "dhw_demand",
        "cooling_demand",
        "heating_demand",
        "indoor_dry_bulb_temperature_cooling_set_point",
        "indoor_dry_bulb_temperature_heating_set_point",
        "comfort_band",
        "hvac_mode",
        "power_outage",
    ]
    for key in exog_keys:
        f[key] = safe_get(current_obs, key, 0.0)

    if exog_override:
        for key, val in exog_override.items():
            f[key] = float(val)

    # Actions at k.
    f["action_electrical_storage"] = float(current_action.get("electrical_storage", 0.0))
    f["action_cooling_device"] = float(current_action.get("cooling_device", 0.0))

    # Current and lagged states/actions.
    for name in STATE_NAMES:
        vals = history.get(name, [])
        f[f"{name}_k"] = float(vals[-1]) if len(vals) else safe_get(current_obs, name, 0.0)
        for lag in lags:
            f[f"{name}_lag_{lag}"] = float(vals[-1 - lag]) if len(vals) > lag else f[f"{name}_k"]

    for name in ["electrical_storage", "cooling_device"]:
        vals = history.get(f"action_{name}", [])
        f[f"action_{name}_k"] = float(current_action.get(name, vals[-1] if vals else 0.0))
        for lag in lags:
            f[f"action_{name}_lag_{lag}"] = float(vals[-1 - lag]) if len(vals) > lag else f[f"action_{name}_k"]

    return f
