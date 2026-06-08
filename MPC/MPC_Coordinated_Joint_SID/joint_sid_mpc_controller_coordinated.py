"""
joint_sid_mpc_controller.py
---------------------------
MPC controller that uses a joint learned SID model for:
    indoor temperature, electrical storage SOC, net electricity, cooling electricity.

This version adds a district-level coordinator ("26th controller") that allocates
the district target dynamically across the 25 local MPC controllers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

from joint_sid_model import (
    JointSIDModel,
    STATE_NAMES,
    normalize_names,
    obs_to_dicts,
    safe_get,
    get_action_bounds,
    build_action_vector,
    build_feature_dict_from_history,
)


@dataclass
class JointSIDMPCConfig:
    horizon: int = 12
    comfort_low: float = 22.0
    comfort_high: float = 26.0

    w_track: float = 100.0
    w_avg_track: float = 500.0
    w_ramp: float = 100.0

    w_comfort: float = 20.0
    w_smooth: float = 50.0
    w_action: float = 5.0
    w_soc: float = 2.0

    terminal_soc_ref: float = 0.50
    maxiter: int = 80
    verbose: bool = False

    use_coordinator: bool = True
    coordinator_eps: float = 1e-6
    min_flex_share: float = 0.02


class JointSIDMPCController:
    """Hierarchical MPC: one district coordinator + local building MPCs.

    Coordinator:
        1. Predicts each building's baseline load using previous action repeated.
        2. Compares district baseline sum with district target.
        3. Allocates the needed correction across buildings using flexibility scores.

    Local MPC:
        Each building tracks its assigned coordinated target trajectory.
    """

    def __init__(
        self,
        env,
        sid_model: JointSIDModel,
        district_target: Sequence[float],
        config: JointSIDMPCConfig | None = None,
    ):
        if minimize is None:
            raise ImportError("scipy is required for MPC optimization.")

        self.env = env
        self.sid = sid_model
        self.target = np.asarray(district_target, dtype=float).ravel()
        self.cfg = config or JointSIDMPCConfig()

        self.n_buildings = len(env.buildings)
        self.obs_names = normalize_names(env, "observation")
        self.action_names = normalize_names(env, "action")
        self.action_bounds = get_action_bounds(env, self.action_names)

        self.prev_actions = np.zeros((self.n_buildings, 2), dtype=float)
        self.histories: List[Dict[str, List[float]]] = [
            self._empty_history() for _ in range(self.n_buildings)
        ]

        self._initialized = False

        self.last_coordinated_targets = None
        self.last_baseline_loads = None
        self.last_flex_weights = None

    def _empty_history(self):
        h = {name: [] for name in STATE_NAMES}
        h["action_electrical_storage"] = []
        h["action_cooling_device"] = []
        return h

    def _init_histories_from_obs(self, obs):
        obs_ds = obs_to_dicts(obs, self.obs_names)

        for i, d in enumerate(obs_ds):
            for name in STATE_NAMES:
                val = safe_get(d, name, 0.0)
                self.histories[i][name] = [val] * (max(self.sid.lags) + 1)

            self.histories[i]["action_electrical_storage"] = [0.0] * (max(self.sid.lags) + 1)
            self.histories[i]["action_cooling_device"] = [0.0] * (max(self.sid.lags) + 1)

        self._initialized = True

    def update_histories_after_step(self, obs, actions):
        obs_ds = obs_to_dicts(obs, self.obs_names)

        for i, d in enumerate(obs_ds):
            for name in STATE_NAMES:
                self.histories[i][name].append(
                    safe_get(d, name, self.histories[i][name][-1])
                )

            act_d = self._action_dict(i, actions[i])
            self.histories[i]["action_electrical_storage"].append(
                float(act_d.get("electrical_storage", 0.0))
            )
            self.histories[i]["action_cooling_device"].append(
                float(act_d.get("cooling_device", 0.0))
            )

            keep = max(self.sid.lags) + 5
            for k in self.histories[i]:
                self.histories[i][k] = self.histories[i][k][-keep:]

    def _action_dict(self, i: int, action_vec) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.action_names[i], action_vec)}

    def _bounds_for_building(self, i: int):
        bds = self.action_bounds[i]
        bat = bds.get("electrical_storage", (0.0, 0.0))
        cool = bds.get("cooling_device", (0.0, 1.0))
        return bat, cool

    def _district_target_horizon(self, t: int):
        H = self.cfg.horizon
        end = min(t + H, len(self.target))
        y = self.target[t:end]

        if len(y) < H:
            pad = y[-1] if len(y) else 0.0
            y = np.r_[y, np.full(H - len(y), pad)]

        return y

    def _time_step(self):
        return int(
            getattr(
                self.env,
                "time_step",
                max(0, len(getattr(self.env, "net_electricity_consumption", [])) - 1),
            )
        )

    def _rollout_building(self, i: int, obs_d: Dict[str, float], h0: Dict[str, List[float]], u_seq: np.ndarray):
        H = self.cfg.horizon
        hist = {k: list(v) for k, v in h0.items()}
        preds = []
        current_obs = dict(obs_d)

        for h in range(H):
            action = {
                "electrical_storage": float(u_seq[h, 0]),
                "cooling_device": float(u_seq[h, 1]),
            }

            feat = build_feature_dict_from_history(
                i,
                current_obs,
                action,
                hist,
                self.sid.lags,
            )

            yhat = self.sid.predict_from_feature_dict(feat)

            T = float(
                np.clip(
                    yhat.get("indoor_dry_bulb_temperature_next", hist["indoor_dry_bulb_temperature"][-1]),
                    10.0,
                    45.0,
                )
            )
            soc = float(
                np.clip(
                    yhat.get("electrical_storage_soc_next", hist["electrical_storage_soc"][-1]),
                    0.0,
                    1.0,
                )
            )
            net = float(
                yhat.get("net_electricity_consumption_next", hist["net_electricity_consumption"][-1])
            )
            cool_e = float(
                max(
                    0.0,
                    yhat.get(
                        "cooling_electricity_consumption_next",
                        hist["cooling_electricity_consumption"][-1],
                    ),
                )
            )

            hist["indoor_dry_bulb_temperature"].append(T)
            hist["electrical_storage_soc"].append(soc)
            hist["net_electricity_consumption"].append(net)
            hist["cooling_electricity_consumption"].append(cool_e)
            hist["action_electrical_storage"].append(action["electrical_storage"])
            hist["action_cooling_device"].append(action["cooling_device"])

            current_obs.update(
                {
                    "indoor_dry_bulb_temperature": T,
                    "electrical_storage_soc": soc,
                    "net_electricity_consumption": net,
                    "cooling_electricity_consumption": cool_e,
                }
            )

            preds.append((T, soc, net, cool_e))

        return np.asarray(preds, dtype=float)

    def _baseline_rollout(self, obs_ds: List[Dict[str, float]]):
        H = self.cfg.horizon
        baseline = np.zeros((self.n_buildings, H), dtype=float)

        for i, obs_d in enumerate(obs_ds):
            u_seq = np.tile(self.prev_actions[i], (H, 1))
            preds = self._rollout_building(i, obs_d, self.histories[i], u_seq)
            baseline[i, :] = preds[:, 2]

        return baseline

    def _building_flex_scores(self, obs_ds: List[Dict[str, float]]):
        up = np.zeros(self.n_buildings, dtype=float)
        down = np.zeros(self.n_buildings, dtype=float)

        for i, obs_d in enumerate(obs_ds):
            bat_b, cool_b = self._bounds_for_building(i)
            bat_low, bat_high = bat_b
            cool_low, cool_high = cool_b

            T = safe_get(obs_d, "indoor_dry_bulb_temperature", 24.0)
            soc = safe_get(obs_d, "electrical_storage_soc", 0.5)

            # Assumption: positive battery action charges battery and increases net load;
            # negative battery action discharges battery and decreases net load.
            battery_up = max(0.0, bat_high) * max(0.0, 1.0 - soc)
            battery_down = max(0.0, -bat_low) * max(0.0, soc)

            cooling_range = max(0.0, cool_high - cool_low)
            comfort_width = max(self.cfg.comfort_high - self.cfg.comfort_low, 1.0)

            # More cooling increases load and is safer when indoor temp is above lower comfort bound.
            cooling_up = cooling_range * max(0.0, T - self.cfg.comfort_low) / comfort_width

            # Less cooling reduces load and is safer when indoor temp is below upper comfort bound.
            cooling_down = cooling_range * max(0.0, self.cfg.comfort_high - T) / comfort_width

            up[i] = battery_up + cooling_up + self.cfg.min_flex_share
            down[i] = battery_down + cooling_down + self.cfg.min_flex_share

        return up, down

    def _coordinated_targets(self, obs_ds: List[Dict[str, float]], t: int):
        H = self.cfg.horizon
        district_target = self._district_target_horizon(t)

        if not self.cfg.use_coordinator:
            return np.tile(district_target / max(self.n_buildings, 1), (self.n_buildings, 1))

        baseline = self._baseline_rollout(obs_ds)
        baseline_district = baseline.sum(axis=0)
        district_error = district_target - baseline_district

        up, down = self._building_flex_scores(obs_ds)

        targets = baseline.copy()
        flex_used = np.zeros((self.n_buildings, H), dtype=float)

        for h in range(H):
            if district_error[h] >= 0.0:
                weights = up / (np.sum(up) + self.cfg.coordinator_eps)
            else:
                weights = down / (np.sum(down) + self.cfg.coordinator_eps)

            targets[:, h] = baseline[:, h] + weights * district_error[h]
            flex_used[:, h] = weights

        self.last_coordinated_targets = targets
        self.last_baseline_loads = baseline
        self.last_flex_weights = flex_used

        return targets

    def _objective_building(
        self,
        flat_u,
        i: int,
        obs_d: Dict[str, float],
        h0: Dict[str, List[float]],
        target_i: np.ndarray,
    ):
        H = self.cfg.horizon
        u = np.asarray(flat_u, dtype=float).reshape(H, 2)

        preds = self._rollout_building(i, obs_d, h0, u)

        T = preds[:, 0]
        soc = preds[:, 1]
        net = preds[:, 2]

        track = np.mean((net - target_i) ** 2)
        avg_track = (np.mean(net) - np.mean(target_i)) ** 2
        ramp = np.mean(np.diff(net) ** 2) if len(net) > 1 else 0.0

        comfort = np.mean(
            np.maximum(self.cfg.comfort_low - T, 0.0) ** 2
            + np.maximum(T - self.cfg.comfort_high, 0.0) ** 2
        )

        du = np.diff(np.vstack([self.prev_actions[i], u]), axis=0)
        smooth = np.mean(du ** 2)

        action_mag = np.mean(u ** 2)
        soc_pen = (soc[-1] - self.cfg.terminal_soc_ref) ** 2

        return float(
            self.cfg.w_track * track
            + self.cfg.w_avg_track * avg_track
            + self.cfg.w_ramp * ramp
            + self.cfg.w_comfort * comfort
            + self.cfg.w_smooth * smooth
            + self.cfg.w_action * action_mag
            + self.cfg.w_soc * soc_pen
        )

    def predict(self, obs):
        if not self._initialized:
            self._init_histories_from_obs(obs)

        obs_ds = obs_to_dicts(obs, self.obs_names)
        t = self._time_step()
        coordinated_targets = self._coordinated_targets(obs_ds, t)

        actions = []

        for i, obs_d in enumerate(obs_ds):
            bat_b, cool_b = self._bounds_for_building(i)
            bounds = [bat_b, cool_b] * self.cfg.horizon

            x0_step = np.array(
                [
                    np.clip(self.prev_actions[i, 0], bat_b[0], bat_b[1]),
                    np.clip(self.prev_actions[i, 1], cool_b[0], cool_b[1]),
                ],
                dtype=float,
            )
            x0 = np.tile(x0_step, self.cfg.horizon)

            res = minimize(
                self._objective_building,
                x0,
                args=(i, obs_d, self.histories[i], coordinated_targets[i]),
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": self.cfg.maxiter,
                    "ftol": 1e-4,
                    "disp": False,
                },
            )

            if not res.success and self.cfg.verbose:
                print(f"Building {i} MPC failed: {res.message}")

            u0 = (
                res.x.reshape(self.cfg.horizon, 2)[0]
                if res.success
                else x0.reshape(self.cfg.horizon, 2)[0]
            )

            self.prev_actions[i] = u0
            actions.append(build_action_vector(self.action_names[i], u0[0], u0[1]))

        return actions
