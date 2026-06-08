"""
sid_mpc_controller.py
---------------------
Control-oriented MPC for CityLearn CE1 using:
  1) learned thermal SID model: Ridge NARX + residual MLP ensemble
  2) approximate analytical battery/SOC dynamics
  3) approximate algebraic electricity balance

Put this file in your repo, for example:
    MPC/MPC_Advanced_SID/sid_mpc_controller.py

Expected SID artifacts from CityLearn_Advanced_SID_Notebook.ipynb:
    saved_files/advanced_sid/sid_preprocessing_and_ridge.joblib
    saved_files/advanced_sid/residual_mlp_seed_0.pt
    saved_files/advanced_sid/residual_mlp_seed_1.pt
    saved_files/advanced_sid/residual_mlp_seed_2.pt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any
import math
import warnings

import joblib
import numpy as np

try:
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover
    minimize = None

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Neural residual model, same architecture as the SID notebook.
# -----------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LearnedThermalSID:
    """Loads the Ridge + residual MLP ensemble SID model."""

    def __init__(self, sid_dir: str | Path, device: str | None = None):
        self.sid_dir = Path(sid_dir)
        meta_path = self.sid_dir / "sid_preprocessing_and_ridge.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Cannot find {meta_path}. Run the advanced SID notebook and save artifacts first."
            )

        bundle = joblib.load(meta_path)
        self.features: List[str] = list(bundle["features"])
        self.lags: Tuple[int, ...] = tuple(bundle.get("lags", (1, 2, 3, 6, 12)))
        self.x_scaler = bundle["x_scaler"]
        self.y_scaler = bundle["y_scaler"]
        self.ridge = bundle["ridge"]

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: List[ResidualMLP] = []
        pt_files = sorted(self.sid_dir.glob("residual_mlp_seed_*.pt"))
        if len(pt_files) == 0:
            warnings.warn("No residual MLP .pt files found; using Ridge-only thermal model.")
        for p in pt_files:
            model = ResidualMLP(n_features=len(self.features), hidden=256, depth=4, dropout=0.05).to(self.device)
            state = torch.load(p, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def _inverse_delta(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

    def predict_delta_T(self, feature_dict: Dict[str, float]) -> float:
        """Predict physical delta T [degC] from a feature dictionary."""
        row = np.array([[float(feature_dict.get(c, 0.0)) for c in self.features]], dtype=np.float32)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.x_scaler.transform(row)

        pred_s = self.ridge.predict(X)
        if self.models:
            Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            residuals = []
            with torch.no_grad():
                for m in self.models:
                    residuals.append(m(Xt).detach().cpu().numpy().ravel()[0])
            pred_s = pred_s + float(np.mean(residuals))

        return float(self._inverse_delta(pred_s)[0])


# -----------------------------------------------------------------------------
# CityLearn helper utilities.
# -----------------------------------------------------------------------------
def normalize_names(env, kind: str = "observation") -> List[List[str]]:
    """Return one list of names per building."""
    n = len(env.buildings)
    raw = env.observation_names if kind == "observation" else env.action_names
    # In this CityLearn fork env.observation_names returns list[list] plus a final 'district_load'.
    if isinstance(raw, list) and len(raw) >= n and isinstance(raw[0], (list, tuple)):
        return [list(x) for x in raw[:n]]
    # Flat fallback.
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
    v = d.get(key, default)
    try:
        if v is None or not np.isfinite(v):
            return default
        return float(v)
    except Exception:
        return default


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


# -----------------------------------------------------------------------------
# MPC configuration.
# -----------------------------------------------------------------------------
@dataclass
class SIDMPCConfig:
    horizon: int = 12
    comfort_low: float = 22.0
    comfort_high: float = 26.0
    w_track: float = 10.0
    w_comfort: float = 150.0
    w_smooth: float = 1.0
    w_soc: float = 0.5
    terminal_soc_ref: float = 0.50
    maxiter: int = 35
    use_slsqp: bool = True
    verbose: bool = False


class SIDMPCController:
    """Decentralized MPC using learned thermal dynamics + known approximate storage/load physics.

    This is intentionally conservative and notebook-friendly. Each building solves a local MPC
    toward an equal share of the district target. That makes it slower but easy to inspect.
    """

    def __init__(self, env, sid_model: LearnedThermalSID, district_target: Sequence[float], config: SIDMPCConfig | None = None):
        if minimize is None:
            raise ImportError("scipy is required for MPC optimization. Install with: pip install scipy")

        self.env = env
        self.sid = sid_model
        self.target = np.asarray(district_target, dtype=float).ravel()
        self.cfg = config or SIDMPCConfig()

        self.n_buildings = len(env.buildings)
        self.obs_names = normalize_names(env, "observation")
        self.action_names = normalize_names(env, "action")
        self.action_bounds = get_action_bounds(env, self.action_names)
        self.prev_actions = np.zeros((self.n_buildings, 2), dtype=float)  # columns: battery, cooling
        self.temperature_histories: List[List[float]] = [[] for _ in range(self.n_buildings)]

    def _bounds_for_building(self, i: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        bds = self.action_bounds[i]
        bat = bds.get("electrical_storage", (0.0, 0.0))
        cool = bds.get("cooling_device", (0.0, 1.0))
        return bat, cool

    def _future_target_share(self, t: int, h: int) -> np.ndarray:
        H = self.cfg.horizon
        end = min(t + H, len(self.target))
        y = self.target[t:end]
        if len(y) < H:
            pad = y[-1] if len(y) else 0.0
            y = np.r_[y, np.full(H - len(y), pad)]
        return y / max(self.n_buildings, 1)

    def _current_timestep(self) -> int:
        return int(getattr(self.env, "time_step", max(0, len(getattr(self.env, "net_electricity_consumption", [])) - 1)))

    def _get_building_array_value(self, building, attr_group: str, attr_name: str, t: int, default: float) -> float:
        try:
            group = getattr(building, attr_group)
            arr = getattr(group, attr_name)
            if arr is None:
                return default
            if t < len(arr):
                val = arr[t]
                if np.isfinite(val):
                    return float(val)
        except Exception:
            pass
        return float(default)

    def _future_base_features(self, i: int, obs_d: Dict[str, float], t_abs: int, h: int, temp: float, soc: float,
                              action_bat: float, action_cool: float, temp_history: List[float]) -> Dict[str, float]:
        """Build feature dictionary in the same convention as the SID notebook."""
        b = self.env.buildings[i]
        t_future = t_abs + h

        hour = safe_get(obs_d, "hour", 1.0)
        month = safe_get(obs_d, "month", 1.0)
        # If future hour/month arrays exist, use them. Otherwise roll hour forward.
        hour = ((hour - 1 + h) % 24) + 1

        outdoor_now = safe_get(obs_d, "outdoor_dry_bulb_temperature", 25.0)
        solar_now = safe_get(obs_d, "direct_solar_irradiance", 0.0)
        outdoor = outdoor_now
        solar_irr = solar_now

        # Use CityLearn-provided 1/2/3-hour forecasts for short horizon where available.
        if h in (1, 2, 3):
            outdoor = safe_get(obs_d, f"outdoor_dry_bulb_temperature_predicted_{h}", outdoor)
            solar_irr = safe_get(obs_d, f"direct_solar_irradiance_predicted_{h}", solar_irr)
        else:
            outdoor = self._get_building_array_value(b, "weather", "outdoor_dry_bulb_temperature", t_future, outdoor)
            solar_irr = self._get_building_array_value(b, "weather", "direct_solar_irradiance", t_future, solar_irr)

        feat = {
            "building_id": float(i),
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "month_sin": math.sin(2 * math.pi * (month - 1) / 12.0),
            "month_cos": math.cos(2 * math.pi * (month - 1) / 12.0),
            "outdoor_dry_bulb_temperature": outdoor,
            "direct_solar_irradiance": solar_irr,
            "outdoor_dry_bulb_temperature_predicted_1": safe_get(obs_d, "outdoor_dry_bulb_temperature_predicted_1", outdoor),
            "outdoor_dry_bulb_temperature_predicted_2": safe_get(obs_d, "outdoor_dry_bulb_temperature_predicted_2", outdoor),
            "outdoor_dry_bulb_temperature_predicted_3": safe_get(obs_d, "outdoor_dry_bulb_temperature_predicted_3", outdoor),
            "direct_solar_irradiance_predicted_1": safe_get(obs_d, "direct_solar_irradiance_predicted_1", solar_irr),
            "direct_solar_irradiance_predicted_2": safe_get(obs_d, "direct_solar_irradiance_predicted_2", solar_irr),
            "direct_solar_irradiance_predicted_3": safe_get(obs_d, "direct_solar_irradiance_predicted_3", solar_irr),
            "indoor_dry_bulb_temperature": temp,
            "indoor_dry_bulb_temperature_cooling_set_point": safe_get(obs_d, "indoor_dry_bulb_temperature_cooling_set_point", self.cfg.comfort_high),
            "indoor_dry_bulb_temperature_heating_set_point": safe_get(obs_d, "indoor_dry_bulb_temperature_heating_set_point", self.cfg.comfort_low),
            "comfort_band": safe_get(obs_d, "comfort_band", self.cfg.comfort_high - self.cfg.comfort_low),
            "hvac_mode": safe_get(obs_d, "hvac_mode", 1.0),
            "power_outage": safe_get(obs_d, "power_outage", 0.0),
            "electrical_storage_soc": soc,
            "non_shiftable_load": safe_get(obs_d, "non_shiftable_load", 0.0),
            "dhw_demand": safe_get(obs_d, "dhw_demand", 0.0),
            "solar_generation": safe_get(obs_d, "solar_generation", 0.0),
            "action_cooling_device": action_cool,
            "action_electrical_storage": action_bat,
        }

        # Update lags from predicted history. lag1 = previous temperature.
        for lag in self.sid.lags:
            if len(temp_history) >= lag:
                feat[f"indoor_dry_bulb_temperature_lag{lag}"] = float(temp_history[-lag])
            else:
                feat[f"indoor_dry_bulb_temperature_lag{lag}"] = float(temp)
        return feat

    def _predict_cooling_electricity(self, building, obs_d: Dict[str, float], action_cool: float, h: int) -> float:
        hvac_mode = safe_get(obs_d, "hvac_mode", 1.0)
        if hvac_mode not in (1.0, 3.0):
            return 0.0
        try:
            nominal = float(building.cooling_device.nominal_power)
            return max(0.0, float(action_cool) * nominal)
        except Exception:
            # fallback to observed current value scaled by action
            return max(0.0, safe_get(obs_d, "cooling_electricity_consumption", 0.0) * max(action_cool, 0.0))

    def _battery_step(self, building, soc: float, action_bat: float, bat_bounds: Tuple[float, float]) -> Tuple[float, float]:
        """Approximate battery physics. Returns (next_soc, electricity_consumption)."""
        try:
            cap = float(building.electrical_storage.capacity)
            eff = float(getattr(building.electrical_storage, "round_trip_efficiency", 0.95))
            loss = float(getattr(building.electrical_storage, "loss_coefficient", 0.0))
        except Exception:
            cap, eff, loss = 1.0, 0.95, 0.0
        cap = max(cap, 1e-6)
        a = float(np.clip(action_bat, bat_bounds[0], bat_bounds[1]))
        energy_balance = a * cap  # CityLearn action convention: fraction of capacity.
        e0 = soc * cap * (1.0 - loss)
        if energy_balance >= 0.0:
            e1 = min(cap, e0 + energy_balance * eff)
        else:
            e1 = max(0.0, e0 + energy_balance / max(eff, 1e-6))
        next_soc = float(np.clip(e1 / cap, 0.0, 1.0))
        return next_soc, float(energy_balance)

    def _base_net_without_control(self, obs_d: Dict[str, float]) -> float:
        """Approximate algebraic base net load excluding cooling and battery."""
        non_shift = safe_get(obs_d, "non_shiftable_load", 0.0)
        dhw_elec = safe_get(obs_d, "dhw_electricity_consumption", 0.0)
        heating_elec = safe_get(obs_d, "heating_electricity_consumption", 0.0)
        solar = safe_get(obs_d, "solar_generation", 0.0)
        return non_shift + dhw_elec + heating_elec - solar

    def _simulate_building_plan(self, i: int, obs_d: Dict[str, float], x: np.ndarray, target_share: np.ndarray, t_abs: int):
        H = self.cfg.horizon
        bat_actions = x[:H]
        cool_actions = x[H:]
        building = self.env.buildings[i]
        bat_bounds, cool_bounds = self._bounds_for_building(i)

        temp = safe_get(obs_d, "indoor_dry_bulb_temperature", 24.0)
        soc = safe_get(obs_d, "electrical_storage_soc", 0.5)
        temp_hist = list(self.temperature_histories[i])[-max(self.sid.lags, default=1):]
        if not temp_hist:
            temp_hist = [temp]

        pred_T = []
        pred_net = []
        pred_soc = []

        base_net = self._base_net_without_control(obs_d)
        prev_bat, prev_cool = self.prev_actions[i]
        cost = 0.0

        for h in range(H):
            a_bat = float(np.clip(bat_actions[h], bat_bounds[0], bat_bounds[1]))
            a_cool = float(np.clip(cool_actions[h], cool_bounds[0], cool_bounds[1]))
            cool_elec = self._predict_cooling_electricity(building, obs_d, a_cool, h)
            soc, bat_elec = self._battery_step(building, soc, a_bat, bat_bounds)

            feat = self._future_base_features(i, obs_d, t_abs, h, temp, soc, a_bat, a_cool, temp_hist)
            dT = self.sid.predict_delta_T(feat)
            temp_next = temp + dT
            # keep numerical sanity; this does not constrain the actual env.
            temp_next = float(np.clip(temp_next, 10.0, 40.0))

            net = base_net + cool_elec + bat_elec
            low_v = max(0.0, self.cfg.comfort_low - temp_next)
            high_v = max(0.0, temp_next - self.cfg.comfort_high)

            cost += self.cfg.w_track * (net - target_share[h]) ** 2
            cost += self.cfg.w_comfort * (low_v ** 2 + high_v ** 2)
            cost += self.cfg.w_smooth * ((a_bat - prev_bat) ** 2 + (a_cool - prev_cool) ** 2)

            pred_T.append(temp_next)
            pred_net.append(net)
            pred_soc.append(soc)
            temp_hist.append(temp_next)
            temp = temp_next
            prev_bat, prev_cool = a_bat, a_cool

        cost += self.cfg.w_soc * (soc - self.cfg.terminal_soc_ref) ** 2
        return cost, np.array(pred_T), np.array(pred_net), np.array(pred_soc)

    def _initial_guess(self, i: int) -> np.ndarray:
        H = self.cfg.horizon
        bat_bounds, cool_bounds = self._bounds_for_building(i)
        # conservative: no battery action, moderate cooling.
        bat0 = np.clip(0.0, bat_bounds[0], bat_bounds[1])
        cool0 = np.clip(self.prev_actions[i, 1], cool_bounds[0], cool_bounds[1])
        return np.r_[np.full(H, bat0), np.full(H, cool0)]

    def _solve_building(self, i: int, obs_d: Dict[str, float], t_abs: int) -> Tuple[float, float, Dict[str, Any]]:
        H = self.cfg.horizon
        target_share = self._future_target_share(t_abs, H)
        bat_bounds, cool_bounds = self._bounds_for_building(i)
        bounds = [bat_bounds] * H + [cool_bounds] * H
        x0 = self._initial_guess(i)

        def obj(x):
            return self._simulate_building_plan(i, obs_d, np.asarray(x), target_share, t_abs)[0]

        try:
            if self.cfg.use_slsqp:
                res = minimize(obj, x0, method="SLSQP", bounds=bounds, options={"maxiter": self.cfg.maxiter, "ftol": 1e-3, "disp": False})
            else:
                res = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": self.cfg.maxiter, "disp": False})
            x = res.x if res.success or hasattr(res, "x") else x0
        except Exception as exc:
            if self.cfg.verbose:
                print(f"Building {i} MPC failed: {exc}")
            x = x0

        x = np.asarray(x, dtype=float)
        cost, T_pred, net_pred, soc_pred = self._simulate_building_plan(i, obs_d, x, target_share, t_abs)
        return float(x[0]), float(x[H]), {"cost": cost, "T_pred": T_pred, "net_pred": net_pred, "soc_pred": soc_pred}

    def predict(self, obs) -> List[List[float]]:
        obs_dicts = obs_to_dicts(obs, self.obs_names)
        t_abs = self._current_timestep()
        actions = []
        for i, obs_d in enumerate(obs_dicts):
            temp_now = safe_get(obs_d, "indoor_dry_bulb_temperature", np.nan)
            if np.isfinite(temp_now):
                self.temperature_histories[i].append(float(temp_now))

            a_bat, a_cool, _ = self._solve_building(i, obs_d, t_abs)
            self.prev_actions[i] = [a_bat, a_cool]
            actions.append(build_action_vector(self.action_names[i], electrical_storage=a_bat, cooling_device=a_cool))
        return actions
