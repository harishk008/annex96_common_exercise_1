"""
run_mpc.py
----------
Main script: Decentralised MPC for all 25 TX buildings.

Pipeline
────────
1. Build the CityLearn TX environment (decentralised mode).
2. Load exogenous data and the district target from CSV files.
3. Instantiate one BuildingMPC controller per building.
4. Simulate the full episode:
     – At each step, each MPC solves its own optimisation (online LSTM
       linearisation, no communication between buildings).
     – The optimal first action is applied to the environment.
5. Compute and display KPIs (NMBE, CV-RMSE, comfort violation rate).
6. Plot results (portfolio tracking, individual building temperature, SOC).

Run from the mpc_claude directory:
    python run_mpc.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Ensure local modules and CityLearn are importable ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))  # mpc_claude/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import (
    make_env,
    get_building_params,
    load_exogenous_data,
    load_district_target,
    compute_cop,
    compute_uncontrollable_net,
    get_sim_window,
)
from mpc_controller import BuildingMPC


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_nmbe(y: np.ndarray, y_ref: np.ndarray) -> float:
    """Normalised Mean Bias Error [%].

    NMBE = mean(y – y_ref) / mean(y_ref) * 100
    Positive = systematic over-consumption; negative = under-consumption.
    """
    return float(np.mean(y - y_ref) / np.mean(y_ref) * 100.0)


def compute_cvrmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    """Coefficient of Variation of RMSE [%].

    CV-RMSE = sqrt(mean((y – y_ref)²)) / mean(y_ref) * 100
    """
    return float(np.sqrt(np.mean((y - y_ref) ** 2)) / np.mean(y_ref) * 100.0)


def comfort_violation_rate(
    temps: np.ndarray,
    setpoints: np.ndarray,
    bands: np.ndarray,
) -> float:
    """Fraction of time steps where the temperature is outside the comfort band [%]."""
    lo  = setpoints - bands
    hi  = setpoints + bands
    vio = np.sum((temps < lo) | (temps > hi))
    return float(vio / len(temps) * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Target split  (proportional to each building's baseline load)
# ─────────────────────────────────────────────────────────────────────────────

def split_target_proportional(
    district_target: np.ndarray,
    uncontrollable: np.ndarray,
) -> np.ndarray:
    """Divide the district target among buildings proportionally to their
    uncontrollable (baseline) net load at each timestep.

    Larger buildings — those with higher baseline consumption after accounting
    for their PV generation — receive a proportionally larger share of the
    district target.  This is fairer than an equal split because it reflects
    each building's actual size and PV capacity.

    Formula at each timestep k:
        weight_i[k]   = uncontrollable_i[k] / Σ_j uncontrollable_j[k]
        target_i[k]   = weight_i[k] * district_target[k]

    If Σ_j uncontrollable_j[k] = 0 (e.g. all PV perfectly offsets loads),
    we fall back to an equal split to avoid division by zero.

    Parameters
    ----------
    district_target : (T,)   array – total portfolio target [kWh]
    uncontrollable  : (T, N) array – per-building uncontrollable net load [kWh]
                      (= non_shiftable + dhw_elec + solar_generation)

    Returns
    -------
    per_building_target : (T, N) array – per-building target share [kWh]
    """
    T, N = uncontrollable.shape

    # Sum of all buildings' baseline loads at each timestep  →  shape (T,)
    total_baseline = uncontrollable.sum(axis=1)   # Σ_j uncontrollable_j[k]

    # Where total baseline is zero, use equal weights to avoid division by zero
    safe_total = np.where(total_baseline == 0, 1.0, total_baseline)  # (T,)

    # Weight of each building at each step  →  shape (T, N)
    weights = uncontrollable / safe_total[:, np.newaxis]

    # Fall back to equal weights (1/N) where total baseline was zero
    equal_weights = np.full((T, N), 1.0 / N)
    weights = np.where(total_baseline[:, np.newaxis] == 0, equal_weights, weights)

    # Per-building target  →  shape (T, N)
    return weights * district_target[:, np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    horizon: int = 24,
    w_track: float = 10.0,
    w_comfort: float = 5.0,
    w_smooth: float = 1.0,
    w_net_smooth: float = 10.0,
    w_terminal_soc: float = 5.0,
    plot: bool = True,
    save_dir: str | None = None,
) -> dict:
    """Run the full decentralised MPC episode and return results.

    Parameters
    ----------
    horizon        : int   MPC prediction horizon H (hours ahead).
    w_track        : float Weight on net-load tracking error.
    w_comfort      : float Weight on temperature comfort slack.
    w_smooth       : float Weight on action smoothness (action-to-action changes).
    w_net_smooth   : float Weight on net-load smoothness (step-to-step net load
                           changes); directly reduces hour-to-hour tracking noise.
    w_terminal_soc : float Weight on terminal SOC cost; prevents the end-of-
                           horizon battery full-cycle oscillation pattern.
    plot           : bool  If True, display and optionally save figures.
    save_dir       : str   Directory to save figures and result CSV; None = skip.

    Returns
    -------
    results : dict with keys:
        portfolio_net     (T,)  – hourly portfolio net consumption [kWh]
        district_target   (T,)  – hourly district target [kWh]
        building_temps    (T, N) – indoor temperature per building [°C]
        building_socs     (T, N) – battery SOC per building [0-1]
        kpis              dict   – NMBE, CV-RMSE, comfort violation rate
    """
    print("=" * 65)
    print(" Decentralised MPC  –  TX Neighbourhood  (25 buildings)")
    print("=" * 65)
    print(f" Horizon  : H = {horizon} steps ({horizon} h)")
    print(f" Weights  : w_track={w_track}, w_comfort={w_comfort}, "
          f"w_smooth={w_smooth}, w_net_smooth={w_net_smooth}, "
          f"w_terminal_soc={w_terminal_soc}")
    print()

    # ── Build environment ─────────────────────────────────────────────────────
    env = make_env(central_agent=False)
    observations, _ = env.reset()

    sim_start, sim_end = get_sim_window(env)
    T = sim_end - sim_start + 1
    N = len(env.buildings)
    H = horizon

    print(f" Simulation : steps {sim_start}–{sim_end}  ({T} hours)")
    print(f" Buildings  : {N}\n")

    # ── Load exogenous data for every building (full window, for forecasting) ─
    exog_list = [
        load_exogenous_data(b, sim_start, sim_end)
        for b in env.buildings
    ]

    # ── District target ───────────────────────────────────────────────────────
    district_target = load_district_target(sim_start, sim_end)   # (T,)

    # ── Pre-compute uncontrollable net load for each building (full window) ───
    unctrllable = np.stack(
        [compute_uncontrollable_net(b, exog_list[i])
         for i, b in enumerate(env.buildings)],
        axis=1,
    )   # shape (T, N)

    # ── Per-building target split (proportional to baseline load) ─────────────
    # Each building receives a share of the district target proportional to its
    # uncontrollable net load — larger buildings get a larger slice.
    target_per_bld = split_target_proportional(district_target, unctrllable)  # (T, N)

    # ── Instantiate one MPC per building ──────────────────────────────────────
    params_list = [get_building_params(b) for b in env.buildings]
    controllers = [
        BuildingMPC(p, horizon=H, w_track=w_track,
                    w_comfort=w_comfort, w_smooth=w_smooth,
                    w_net_smooth=w_net_smooth,
                    w_terminal_soc=w_terminal_soc)
        for p in params_list
    ]

    # Initialise each controller's prev_net_total to the first step's actual
    # uncontrollable load (instead of 0.0) so the net-load smoothness penalty
    # does not cause an artificial spike on the very first solve call.
    for i, ctrl in enumerate(controllers):
        ctrl.prev_net_total = float(unctrllable[0, i])

    print(" MPC controllers initialised.  Starting episode …\n")

    # ── Storage for logging ───────────────────────────────────────────────────
    portfolio_net  = np.zeros(T)          # total net consumption all buildings
    building_temps = np.zeros((T, N))     # indoor temperature per building
    building_socs  = np.zeros((T, N))     # battery SOC per building
    building_nets  = np.zeros((T, N))     # net electricity per building
    solve_statuses = []                   # CVXPY solve status per step

    # actual_T tracks how many steps were truly simulated.  The CityLearn env
    # signals terminated=True one or two steps before the full T, so the
    # initialised-zero tail of the arrays must be trimmed before KPIs / plots.
    actual_T = T

    # Keep track of previous temperature and cooling demand for disturbance est.
    prev_temps   = [None] * N
    prev_cd      = [None] * N

    from tqdm import tqdm
    for step in tqdm(range(T), desc="MPC", unit="step", dynamic_ncols=True):
        abs_step = sim_start + step        # absolute time index

        # ── Extract current observations from the env ─────────────────────
        # observations is a list of N observation vectors (decentralised mode)
        # We read the values we need directly from the building objects for
        # clarity (avoids indexing into the raw observation vector).
        temps = np.array([b.observations()["indoor_dry_bulb_temperature"]
                          for b in env.buildings])          # (N,)
        socs  = np.array([b.observations()["electrical_storage_soc"]
                          for b in env.buildings])          # (N,)

        # ── Build horizon-length forecasts ────────────────────────────────
        # Clamp to the last available data point if near the episode end.
        h_end = min(step + H, T)
        h_len = h_end - step               # actual usable forecast length

        def _pad(arr):
            """Pad a short array to length H by repeating the last value."""
            if len(arr) < H:
                return np.concatenate([arr, np.full(H - len(arr), arr[-1])])
            return arr

        statuses_step = []
        actions_step  = []

        for i, (building, ctrl, exog_df) in enumerate(
            zip(env.buildings, controllers, exog_list)
        ):
            # Forecast slices (absolute indices → relative to exog_df index)
            fcast_idx = range(abs_step, abs_step + H)
            # Clamp to valid range
            fcast_idx_clamped = [min(j, sim_end) for j in fcast_idx]

            t_outdoor_fcast  = np.array(
                [exog_df.loc[j, "outdoor_dry_bulb_temperature"]
                 for j in fcast_idx_clamped]
            )
            unctrll_fcast    = np.array(
                [unctrllable[min(step + jj, T - 1), i]
                 for jj in range(H)]
            )
            target_fcast     = np.array(
                [target_per_bld[min(step + jj, T - 1), i]
                 for jj in range(H)]
            )
            # Annex96 challenge defines a fixed thermal comfort window for the
            # cooling season: 22–26 °C.  We use a midpoint setpoint of 24 °C
            # and a half-band of 2 °C, giving exactly [22, 26] °C.
            # (The per-building CSV setpoints are NOT used here because they
            # vary with occupancy schedules and would produce a shifting band.)
            comfort_sp   = 24.0   # midpoint of [22, 26] °C
            comfort_band =  2.0   # half-width → lower=22, upper=26 °C

            # ── Solve MPC ─────────────────────────────────────────────────
            u_opt, info = ctrl.solve(
                building           = building,
                temp_obs           = float(temps[i]),
                soc_obs            = float(socs[i]),
                target_per_building= target_fcast,
                uncontrollable_net = unctrll_fcast,
                outdoor_temp_forecast = t_outdoor_fcast,
                comfort_setpoint   = comfort_sp,
                comfort_band       = comfort_band,
                prev_temp          = prev_temps[i],
                prev_cooling_demand= prev_cd[i],
            )

            actions_step.append(list(u_opt))   # [u_battery, u_cooling]
            statuses_step.append(info["status"])

            # Update previous-step memory for disturbance estimation
            prev_temps[i] = float(temps[i])
            # Cooling demand this step: u_cool * P_hp * COP
            cop_now = float(compute_cop(building,
                                        np.array([exog_df.loc[abs_step,
                                        "outdoor_dry_bulb_temperature"]]))[0])
            prev_cd[i]   = float(u_opt[1] * ctrl.P_hp * cop_now)

        # ── Step environment ──────────────────────────────────────────────
        observations, rewards, terminated, truncated, info_env = env.step(
            actions_step
        )

        # ── Log results AFTER the step ────────────────────────────────────
        # After env.step(), b.net_electricity_consumption[-1] is the net load
        # produced by the action we just applied (most recent completed step).
        for i, b in enumerate(env.buildings):
            net_i = float(b.net_electricity_consumption[-1])
            building_nets[step, i]  = net_i
            portfolio_net[step]    += net_i
            # Temperature and SOC: read from the observations dict which uses
            # the current (post-step) time_step value.
            obs_i = b.observations()
            building_temps[step, i] = obs_i["indoor_dry_bulb_temperature"]
            building_socs[step, i]  = obs_i["electrical_storage_soc"]

        solve_statuses.append(statuses_step)

        # ── Update tqdm postfix every step with live stats ────────────────
        opt_pct = 100 * sum(
            1 for s in statuses_step if "optimal" in str(s)
        ) / max(1, N)
        tqdm.write("") if False else None   # no-op; keeps reference in scope
        # Access the tqdm bar via the loop variable name trick
        # (tqdm is imported above; we write stats as postfix on the bar)
        # The bar object is bound to the for-loop iterator — update via the
        # loop's tqdm wrapper (captured as the iterable's internal state).
        # Since we can't reference the bar directly inside the loop, we use
        # a lightweight workaround: write postfix info every 24 steps via
        # tqdm.write so it appears above the bar without garbling it.
        if (step + 1) % 24 == 0 or step == T - 1:
            tqdm.write(
                f"  step {step+1:4d}/{T} | "
                f"portfolio={portfolio_net[max(0,step-23):step+1].mean():.2f} kWh | "
                f"target={district_target[max(0,step-23):step+1].mean():.2f} kWh | "
                f"opt={opt_pct:.0f}%"
            )

        if terminated or truncated:
            actual_T = step + 1   # record how many steps actually completed
            break

    # Trim all logging arrays to the steps that were actually simulated.
    # The CityLearn env terminates slightly before the full T window so the
    # remaining rows would be zero-padded, causing fake spikes in the plots
    # and inflated comfort-violation counts.
    portfolio_net  = portfolio_net[:actual_T]
    building_temps = building_temps[:actual_T]
    building_socs  = building_socs[:actual_T]
    building_nets  = building_nets[:actual_T]
    district_target_trimmed = district_target[:actual_T]

    print(f"\n Completed {actual_T}/{T} steps"
          + ("  (episode ended early)" if actual_T < T else ""))

    # ─────────────────────────────────────────────────────────────────────────
    # KPIs
    # ─────────────────────────────────────────────────────────────────────────

    # Use only steps where the target is non-zero (meaningful tracking window)
    valid = district_target_trimmed > 0.0
    y     = portfolio_net[valid]
    y_ref = district_target_trimmed[valid]

    nmbe   = compute_nmbe(y, y_ref)
    cvrmse = compute_cvrmse(y, y_ref)

    # Comfort: check against the fixed Annex96 window [22, 26] °C
    # across all buildings and all steps.
    all_temps     = building_temps.reshape(-1)
    all_setpoints = np.full_like(all_temps, 24.0)   # fixed midpoint
    all_bands     = np.full_like(all_temps,  2.0)   # fixed half-width

    comfort_vio = comfort_violation_rate(all_temps, all_setpoints, all_bands)

    kpis = {
        "NMBE [%]":              round(nmbe,    3),
        "CV-RMSE [%]":           round(cvrmse,  3),
        "Comfort violation [%]": round(comfort_vio, 2),
    }

    print("\n" + "=" * 50)
    print(" KPIs  (tracking district load target)")
    print("=" * 50)
    for k, v in kpis.items():
        print(f"  {k:30s} : {v:8.3f}")
    print("=" * 50 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────

    if plot:
        _make_plots(
            portfolio_net    = portfolio_net,
            district_target  = district_target_trimmed,
            building_temps   = building_temps,
            building_socs    = building_socs,
            T                = actual_T,
            save_dir         = save_dir,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Optional CSV export
    # ─────────────────────────────────────────────────────────────────────────

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "portfolio_net":    portfolio_net,
            "district_target":  district_target_trimmed,
        }).to_csv(out / "portfolio_tracking.csv", index_label="step")

        pd.DataFrame(building_nets,
                     columns=[b.name for b in env.buildings]).to_csv(
            out / "building_net_consumption.csv", index_label="step"
        )
        pd.DataFrame(building_temps,
                     columns=[b.name for b in env.buildings]).to_csv(
            out / "building_temperatures.csv", index_label="step"
        )
        pd.DataFrame(kpis, index=[0]).to_csv(out / "kpis.csv", index=False)
        print(f" Results saved to {out}/")

    return {
        "portfolio_net":   portfolio_net,
        "district_target": district_target_trimmed,
        "building_temps":  building_temps,
        "building_socs":   building_socs,
        "kpis":            kpis,
        "actual_T":        actual_T,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_plots(
    portfolio_net,
    district_target,
    building_temps,
    building_socs,
    T,
    save_dir,
):
    steps = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Decentralised MPC  –  TX Neighbourhood", fontsize=13)

    # ── Plot 1: Portfolio load vs district target ─────────────────────────────
    ax = axes[0]
    ax.plot(steps, district_target, "k--",  lw=1.5, label="District target")
    ax.plot(steps, portfolio_net,   "b-",   lw=1.2, label="Portfolio net load")
    ax.fill_between(steps, district_target, portfolio_net,
                    alpha=0.15, color="blue")
    ax.set_ylabel("Net load [kWh]")
    ax.set_title("Portfolio load tracking")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Building indoor temperatures (all 25, faded + mean) ──────────
    ax = axes[1]
    N  = building_temps.shape[1]
    for i in range(N):
        ax.plot(steps, building_temps[:, i], color="steelblue", alpha=0.2, lw=0.7)
    ax.plot(steps, building_temps.mean(axis=1), "b-", lw=1.8, label="Mean temp")
    # Fixed Annex96 comfort window: 22–26 °C (cooling season)
    ax.axhline(26.0, color="red", ls="--", lw=1.2, label="Comfort band (22–26 °C)")
    ax.axhline(22.0, color="red", ls="--", lw=1.2)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Indoor temperatures  (all buildings)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Battery SOC (all 25, faded + mean) ────────────────────────────
    ax = axes[2]
    for i in range(N):
        ax.plot(steps, building_socs[:, i], color="darkorange", alpha=0.2, lw=0.7)
    ax.plot(steps, building_socs.mean(axis=1), "-",
            color="darkorange", lw=1.8, label="Mean SOC")
    ax.set_ylabel("Battery SOC [–]")
    ax.set_xlabel("Simulation step (hours)")
    ax.set_title("Battery state of charge  (all buildings)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(save_dir) / "mpc_results.png"), dpi=150)
        print(f" Figure saved → {save_dir}/mpc_results.png")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = str(Path(__file__).parent / "saved_files" / f"mpc_claude_run_{timestamp}")
    results = run(
        horizon        = 36,
        w_track        = 100.0,
        w_comfort      = 5.0,
        w_smooth       = 100.0,
        w_net_smooth   = 100.0,
        w_terminal_soc = 5.0,
        plot           = True,
        save_dir       = save_directory,
    )
