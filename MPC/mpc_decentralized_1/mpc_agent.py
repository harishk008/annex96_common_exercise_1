"""
mpc_agent.py
------------
Thin wrapper that gives the MPC a predict(observations) interface
matching the CityLearn RBC agents, so it can be used in the same
simulation loop pattern:

    agent = MPCAgent(env)
    while not env.terminated:
        actions = agent.predict(obs)
        obs, _, _, _, _ = env.step(actions)

The env passed here must have been created with central_agent=False
(the MPC is decentralised — one controller per building).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# ── Make sure local modules are importable ────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

from utils import (
    get_building_params,
    load_exogenous_data,
    load_district_target,
    compute_cop,
    compute_uncontrollable_net,
    get_sim_window,
)
from mpc_controller import BuildingMPC
from run_mpc import split_target_proportional


class MPCAgent:
    """Decentralised MPC agent with a CityLearn-compatible predict() interface.

    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment created with ``central_agent=False``.
        Must have been reset before passing here (``env.reset()``).
    horizon : int
        MPC prediction horizon in hours.
    w_track : float
        Weight on net-load tracking error.
    w_comfort : float
        Weight on comfort slack (soft constraint).
    w_smooth : float
        Weight on action smoothness.
    w_net_smooth : float
        Weight on net-load step-to-step smoothness.
    w_terminal_soc : float
        Weight on terminal SOC cost (prevents end-of-horizon battery cycling).

    Usage
    -----
    >>> env = make_env(central_agent=False)
    >>> obs, _ = env.reset()
    >>> agent = MPCAgent(env)
    >>> while not env.terminated:
    ...     actions = agent.predict(obs)
    ...     obs, _, _, _, _ = env.step(actions)
    """

    def __init__(
        self,
        env,
        horizon: int = 24,
        w_track: float = 10.0,
        w_comfort: float = 5.0,
        w_smooth: float = 1.0,
        w_net_smooth: float = 10.0,
        w_terminal_soc: float = 5.0,
    ):
        self.env = env
        self._step = 0                       # internal timestep counter

        sim_start, sim_end = get_sim_window(env)
        self._sim_start = sim_start
        self._T         = sim_end - sim_start + 1
        N               = len(env.buildings)

        # ── Pre-load all exogenous data (full horizon) ────────────────────────
        self._exog_list = [
            load_exogenous_data(b, sim_start, sim_end)
            for b in env.buildings
        ]

        # ── District target (full horizon) ────────────────────────────────────
        district_target = load_district_target(sim_start, sim_end)  # (T,)

        # ── Uncontrollable net load per building (full horizon) ───────────────
        self._unctrllable = np.stack(
            [compute_uncontrollable_net(b, self._exog_list[i])
             for i, b in enumerate(env.buildings)],
            axis=1,
        )   # (T, N)

        # ── Per-building target split (proportional to baseline load) ─────────
        self._target_per_bld = split_target_proportional(
            district_target, self._unctrllable
        )   # (T, N)

        # ── Create one MPC controller per building ────────────────────────────
        params_list = [get_building_params(b) for b in env.buildings]
        self._controllers = [
            BuildingMPC(
                p, horizon=horizon,
                w_track=w_track, w_comfort=w_comfort,
                w_smooth=w_smooth, w_net_smooth=w_net_smooth,
                w_terminal_soc=w_terminal_soc,
            )
            for p in params_list
        ]

        # Initialise net-load smoothness memory to first-step uncontrollable
        # (avoids an artificial spike at step 0)
        for i, ctrl in enumerate(self._controllers):
            ctrl.prev_net_total = float(self._unctrllable[0, i])

        # ── State tracking for disturbance estimation ─────────────────────────
        self._prev_temps = [None] * N
        self._prev_cd    = [None] * N

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, observations) -> list[list[float]]:
        """Compute optimal actions for the current timestep.

        Parameters
        ----------
        observations : list
            Current observations from env.step() / env.reset() — not used
            directly (state is read from building objects for precision),
            but kept for API compatibility with CityLearn RBC agents.

        Returns
        -------
        actions : list[list[float]]
            List of [u_battery, u_cooling] per building, ready to pass to
            env.step().
        """
        step     = self._step
        abs_step = self._sim_start + step
        H        = self._controllers[0].H

        # Read current state from building objects (more precise than raw obs)
        temps = np.array([
            b.observations()["indoor_dry_bulb_temperature"]
            for b in self.env.buildings
        ])
        socs = np.array([
            b.observations()["electrical_storage_soc"]
            for b in self.env.buildings
        ])

        actions = []
        for i, (building, ctrl, exog_df) in enumerate(
            zip(self.env.buildings, self._controllers, self._exog_list)
        ):
            # Clamp forecast indices to episode end
            fcast_idx = [min(abs_step + j, self._sim_start + self._T - 1)
                         for j in range(H)]

            t_outdoor_fcast = np.array([
                exog_df.loc[j, "outdoor_dry_bulb_temperature"]
                for j in fcast_idx
            ])
            unctrll_fcast = np.array([
                self._unctrllable[min(step + j, self._T - 1), i]
                for j in range(H)
            ])
            target_fcast = np.array([
                self._target_per_bld[min(step + j, self._T - 1), i]
                for j in range(H)
            ])

            u_opt, _ = ctrl.solve(
                building              = building,
                temp_obs              = float(temps[i]),
                soc_obs               = float(socs[i]),
                target_per_building   = target_fcast,
                uncontrollable_net    = unctrll_fcast,
                outdoor_temp_forecast = t_outdoor_fcast,
                comfort_setpoint      = 24.0,
                comfort_band          = 2.0,
                prev_temp             = self._prev_temps[i],
                prev_cooling_demand   = self._prev_cd[i],
            )
            actions.append(list(u_opt))

            # Update disturbance estimation memory
            self._prev_temps[i] = float(temps[i])
            cop_now = float(compute_cop(
                building,
                np.array([exog_df.loc[abs_step, "outdoor_dry_bulb_temperature"]])
            )[0])
            self._prev_cd[i] = float(u_opt[1] * ctrl.P_hp * cop_now)

        self._step += 1
        return actions
