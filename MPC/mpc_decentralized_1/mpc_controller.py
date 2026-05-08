"""
mpc_controller.py
-----------------
Per-building decentralised Linear MPC controller.

State for building i:
    x_i[k] = [indoor_dry_bulb_temperature_i[k],   (°C)
               electrical_storage_soc_i[k]]        (fraction 0–1)

Actions for building i (TX dataset):
    u_i[k] = [u_battery_i[k],   ∈ [action_low[0], action_high[0]]
               u_cooling_i[k]]  ∈ [0, 1]

Linearised dynamics (updated every step via online LSTM Jacobians):
─────────────────────────────────────────────────────────────────────
  temp[k+j+1] = a_j * temp[k+j]
                + (b_j * nominal_power_hp * COP[k+j]) * u_cooling[k+j]
                + d_temp[k+j]

  soc[k+j+1]  = soc[k+j]
                + (nominal_power_bat * efficiency / capacity) * u_battery[k+j]

where:
  a_j, b_j   = per-step LSTM Jacobians (refreshed online each call)
  COP[k+j]   = heat-pump COP computed from forecast outdoor temperature
  d_temp[k]  = disturbance estimate (exogenous effects)  – held constant over H

Controllable net electricity consumption (per building):
────────────────────────────────────────────────────────
  net_ctrl[k] = u_cooling[k] * nominal_power_hp       (HVAC electrical draw)
              + u_battery[k] * nominal_power_bat       (battery charge, –ve = discharge)

  net_total[k] = net_ctrl[k]  +  uncontrollable[k]
               uncontrollable = non_shiftable + dhw_elec – solar  (from CSV)

MPC objective:
──────────────
  min   Σ_{j=0}^{H-1}  w_track    * (net_total_i[k+j] – target_i[k+j])²
                      + w_comfort  * (slack_hot[j]² + slack_cold[j]²)
                      + w_smooth   * ||u[j] – u[j-1]||²
                      + w_net_smooth * (net_total_i[k+j] – net_total_i[k+j-1])²

  s.t.  linearised dynamics (above)
        action bounds            u_low ≤ u[j] ≤ u_high
        SOC bounds               (1 – DOD) ≤ soc[j] ≤ 1
        comfort (soft)           T_cool_setpoint – comfort_band – slack_cold ≤ temp
                                 temp ≤ T_cool_setpoint + comfort_band + slack_hot
                                 slack_hot, slack_cold ≥ 0
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from system_identification import get_online_jacobians
from utils import compute_cop


# ─────────────────────────────────────────────────────────────────────────────
# BuildingMPC
# ─────────────────────────────────────────────────────────────────────────────

class BuildingMPC:
    """Decentralised linear MPC for one CityLearn building.

    Parameters
    ----------
    params : dict
        Output of utils.get_building_params() for this building.
    horizon : int
        Prediction and control horizon H (number of time steps).
    w_track : float
        Weight on net-consumption tracking error (primary objective).
    w_comfort : float
        Weight on temperature comfort violation (soft constraint slack).
    w_smooth : float
        Weight on action smoothness (penalises large action changes).
    w_net_smooth : float
        Weight on net-load smoothness (penalises large step-to-step swings in
        net electricity consumption — directly targets tracking noise).
    w_terminal_soc : float
        Weight on the terminal SOC cost.  At the end of the horizon the MPC
        penalises (soc[H] − 0.5)² so the battery is not driven to extremes
        within every horizon window — this is the primary fix for the
        "end-of-horizon" full-cycle battery oscillation.
    """

    def __init__(
        self,
        params: dict,
        horizon: int = 6,
        w_track: float = 10.0,
        w_comfort: float = 5.0,
        w_smooth: float = 1.0,
        w_net_smooth: float = 10.0,
        w_terminal_soc: float = 5.0,
    ):
        self.params           = params
        self.H                = horizon
        self.w_track          = w_track
        self.w_comfort        = w_comfort
        self.w_smooth         = w_smooth
        self.w_net_smooth     = w_net_smooth
        self.w_terminal_soc   = w_terminal_soc

        # Physical device parameters (fixed per building)
        self.P_hp  = params["nominal_power_hp"]    # kW (electrical)
        self.P_bat = params["nominal_power_bat"]   # kW
        self.C_bat = params["battery_capacity"]    # kWh
        self.eta   = params["battery_efficiency"]  # round-trip eff.
        self.dod   = params["depth_of_discharge"]  # max discharge fraction
        self.soc_min = 1.0 - self.dod              # lower SOC bound

        # Battery SOC update coefficient  (per time step, 1 h)
        # soc[k+1] = soc[k] + alpha_bat * u_battery[k]
        self.alpha_bat = self.P_bat * self.eta / self.C_bat

        # Action bounds [u_battery_low, u_battery_high],
        #               [u_cooling_low, u_cooling_high]
        self.u_low  = params["action_low"]    # shape (2,)
        self.u_high = params["action_high"]   # shape (2,)

        # Previous action (for smoothness penalty); initialised to zero
        self.u_prev = np.zeros(2)

        # Previous net-total electricity consumption [kWh] (for net-load
        # smoothness penalty); initialised to zero — updated every call.
        self.prev_net_total = 0.0

        # Disturbance estimate (exogenous effect on temperature, °C / step)
        # Held constant over the horizon; updated each call via feedback.
        self.d_temp = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def solve(
        self,
        building,
        temp_obs: float,
        soc_obs: float,
        target_per_building: np.ndarray,
        uncontrollable_net: np.ndarray,
        outdoor_temp_forecast: np.ndarray,
        comfort_setpoint: float,
        comfort_band: float,
        prev_temp: float | None = None,
        prev_cooling_demand: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Solve the MPC optimisation for the next H steps.

        Parameters
        ----------
        building : LSTMDynamicsBuilding
            Live building object (used for online LSTM Jacobians and COP).
        temp_obs : float
            Current observed indoor temperature [°C].
        soc_obs : float
            Current observed battery state of charge [0–1].
        target_per_building : np.ndarray, shape (H,)
            Per-building share of the district target [kWh] for steps k…k+H-1.
        uncontrollable_net : np.ndarray, shape (H,)
            Non-controllable net electricity [kWh] for steps k…k+H-1.
        outdoor_temp_forecast : np.ndarray, shape (H,)
            Forecast outdoor dry-bulb temperature [°C] for steps k…k+H-1.
        comfort_setpoint : float
            Cooling setpoint temperature [°C].
        comfort_band : float
            Allowed deviation from the setpoint [°C].
        prev_temp : float or None
            Observed temperature one step ago (used for disturbance update).
        prev_cooling_demand : float or None
            Actual cooling demand one step ago [kWh thermal] (for disturbance).

        Returns
        -------
        u_opt : np.ndarray, shape (2,)
            Optimal first action [u_battery, u_cooling] to apply now.
        info : dict
            Diagnostic information (predicted trajectories, solve status).
        """
        H = self.H

        # ── Step 1: Get online LSTM Jacobians (a, b_phys) ──────────────────
        # a_phys : ∂temp / ∂temp_prev      (dimensionless)
        # b_phys : ∂temp / ∂cooling_demand [°C / kWh_thermal]
        a_phys, b_phys = get_online_jacobians(building)

        # ── Step 2: Compute COP forecast over horizon ───────────────────────
        # The effective control gain for temperature is:
        #   B_eff[j] = b_phys * P_hp * COP[k+j]
        # because cooling_demand = u_cooling * P_hp * COP (thermal kWh)
        cops = compute_cop(building, outdoor_temp_forecast[:H])  # shape (H,)
        B_eff = b_phys * self.P_hp * cops                         # shape (H,)

        # ── Step 3: Update disturbance estimate from last step's feedback ──
        # d_temp estimates ALL exogenous temperature effects not captured by
        # the linearised model.  We compute it from the actual observation:
        #   d_hat[k] = temp[k] - a * temp[k-1] - b * cooling_demand[k-1]
        # Then propagate as constant over the horizon.
        if prev_temp is not None and prev_cooling_demand is not None:
            self.d_temp = (
                temp_obs
                - a_phys * prev_temp
                - b_phys * prev_cooling_demand
            )

        # ── Step 4: Build CVXPY optimisation problem ────────────────────────

        # Decision variables
        u = cp.Variable((H, 2))          # u[:, 0] = u_battery,  u[:, 1] = u_cooling
        temp      = cp.Variable(H + 1)   # predicted temperature trajectory
        soc       = cp.Variable(H + 1)   # predicted SOC trajectory
        slack_hot  = cp.Variable(H, nonneg=True)  # temp above comfort upper bound
        slack_cold = cp.Variable(H, nonneg=True)  # temp below comfort lower bound

        constraints = []
        cost = 0.0

        # ── Initial state (observation feedback) ────────────────────────────
        constraints += [
            temp[0] == temp_obs,
            soc[0]  == soc_obs,
        ]

        # ── Effective SOC lower bound ─────────────────────────────────────────
        # If the current observed SOC is already below soc_min (e.g. battery
        # starts empty but DOD < 100%), we cannot enforce the hard lower bound
        # for the IMMEDIATE next step, because the max charge rate may be too
        # small to reach soc_min in one step.  We clamp the effective lower
        # bound to the current SOC so the problem stays feasible, and let the
        # optimizer charge the battery back up as quickly as possible.
        effective_soc_min = min(self.soc_min, float(soc_obs))

        for j in range(H):
            u_bat    = u[j, 0]  # battery action  (– = discharge, + = charge)
            u_cool   = u[j, 1]  # cooling device  (0 = off, 1 = full)

            # ── Temperature dynamics (linearised LSTM) ──────────────────────
            # temp[j+1] = a * temp[j] + B_eff[j] * u_cool[j] + d_temp
            # Note: b_phys is typically negative (cooling reduces temp),
            # so B_eff = b_phys * P_hp * COP < 0.
            constraints += [
                temp[j + 1] == a_phys * temp[j]
                              + B_eff[j] * u_cool
                              + self.d_temp
            ]

            # ── Battery SOC dynamics (exact analytical model) ───────────────
            # soc[j+1] = soc[j] + alpha_bat * u_battery[j]
            constraints += [
                soc[j + 1] == soc[j] + self.alpha_bat * u_bat
            ]

            # ── Action bounds ───────────────────────────────────────────────
            constraints += [
                u_bat  >= self.u_low[0],
                u_bat  <= self.u_high[0],
                u_cool >= self.u_low[1],
                u_cool <= self.u_high[1],
            ]

            # ── SOC bounds ───────────────────────────────────────────────────
            # Lower bound uses effective_soc_min (≤ soc_min) so that when the
            # battery starts below the depth-of-discharge limit, the problem
            # remains feasible; the optimizer will charge as fast as possible.
            constraints += [
                soc[j + 1] >= effective_soc_min,
                soc[j + 1] <= 1.0,
            ]

            # ── Soft comfort constraints (slack variables) ───────────────────
            # Comfort window: [setpoint – band, setpoint + band]
            T_lo = comfort_setpoint - comfort_band
            T_hi = comfort_setpoint + comfort_band
            constraints += [
                temp[j + 1] >= T_lo - slack_cold[j],
                temp[j + 1] <= T_hi + slack_hot[j],
            ]

            # ── Net electricity consumption (controllable part) ──────────────
            # net_ctrl[j] = u_cool * P_hp  (HVAC electrical draw, ≥ 0)
            #             + u_bat  * P_bat (battery: + = charging, – = discharging)
            net_ctrl  = u_cool * self.P_hp + u_bat * self.P_bat
            net_total = net_ctrl + uncontrollable_net[j]

            # ── Tracking cost ─────────────────────────────────────────────────
            track_err = net_total - target_per_building[j]
            cost += self.w_track * cp.square(track_err)

            # ── Comfort cost ──────────────────────────────────────────────────
            cost += self.w_comfort * (
                cp.square(slack_hot[j]) + cp.square(slack_cold[j])
            )

            # ── Action smoothness cost ────────────────────────────────────────
            # Penalise large changes in u_battery and u_cooling between steps.
            if j == 0:
                u_prev_j = self.u_prev          # compare to last applied action
            else:
                u_prev_j = u[j - 1, :]
            cost += self.w_smooth * cp.sum_squares(u[j, :] - u_prev_j)

            # ── Net-load smoothness cost ──────────────────────────────────────
            # Penalise large swings in net electricity consumption between
            # consecutive steps. This directly reduces tracking noise by
            # discouraging the controller from abruptly charging/discharging
            # or toggling HVAC between time steps.
            #   j = 0 : compare against the net load from the last real step
            #   j > 0 : compare against the predicted net load at step j-1
            if j == 0:
                # net_total_prev is the actual net load from the previous step
                # (stored as self.prev_net_total, updated after each solve)
                net_prev = self.prev_net_total
            else:
                # net_ctrl for step j-1 using the previous loop iteration's u
                net_prev = (u[j - 1, 1] * self.P_hp
                            + u[j - 1, 0] * self.P_bat
                            + uncontrollable_net[j - 1])
            cost += self.w_net_smooth * cp.square(net_total - net_prev)

        # ── Terminal SOC cost ────────────────────────────────────────────────
        # Penalise deviation from the mid-point SOC (0.5) at the end of the
        # horizon.  Without this, MPC with a finite horizon tends to drain or
        # fill the battery completely within every horizon window and then
        # repeat the pattern — producing large, periodic net-load oscillations.
        # Keeping soc[H] near 0.5 spreads capacity evenly for future steps.
        cost += self.w_terminal_soc * cp.square(soc[H] - 0.5)

        # ── Solve ────────────────────────────────────────────────────────────
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_starting=True, verbose=False)
        except cp.error.SolverError:
            problem.solve(solver=cp.SCS, verbose=False)

        # ── Extract solution ──────────────────────────────────────────────────
        if u.value is not None and problem.status in (
            "optimal", "optimal_inaccurate"
        ):
            u_opt = u.value[0, :]    # first step of optimal sequence

            # Clip to hard bounds (numerical safety)
            u_opt[0] = float(np.clip(u_opt[0], self.u_low[0], self.u_high[0]))
            u_opt[1] = float(np.clip(u_opt[1], self.u_low[1], self.u_high[1]))
        else:
            # Fallback: repeat previous action clipped to bounds
            u_opt = np.clip(self.u_prev.copy(), self.u_low, self.u_high)

        # Update memory of previous action
        self.u_prev = u_opt.copy()

        # Update previous net-total for the next call's smoothness penalty.
        # We use the actual applied u_opt (not the predicted trajectory) so
        # that the disturbance feedback and the net-load memory stay in sync.
        self.prev_net_total = (
            float(u_opt[1]) * self.P_hp
            + float(u_opt[0]) * self.P_bat
            + float(uncontrollable_net[0])
        )

        # Diagnostic info
        info = {
            "status":      problem.status,
            "cost":        float(problem.value) if problem.value is not None else np.nan,
            "a_phys":      a_phys,
            "b_phys":      b_phys,
            "d_temp":      self.d_temp,
            "temp_pred":   temp.value if temp.value is not None else None,
            "soc_pred":    soc.value  if soc.value  is not None else None,
        }

        return u_opt, info
