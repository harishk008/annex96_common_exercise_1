"""
utils.py
--------
Helper utilities shared across system identification and MPC modules.

Covers:
  - CityLearn environment construction
  - Building physical parameter extraction
  - Exogenous data loading (weather, loads, solar, target)
  - COP and uncontrollable-net computation helpers
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo path setup ───────────────────────────────────────────────────────────
# This file lives at:  annex96_common_exercise_1/MPC/ALMPC/utils.py
# REPO_ROOT is two levels up: annex96_common_exercise_1/
REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = REPO_ROOT / "data" / "datasets" / "annex96_ce1_tx_neighborhood"
SCHEMA_PATH = DATA_DIR / "schema.json"

# Make sure the local CityLearn copy is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from citylearn.citylearn import CityLearnEnv


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(central_agent: bool = False) -> CityLearnEnv:
    """Create a fresh TX CityLearn environment.

    Parameters
    ----------
    central_agent : bool
        False  → decentralised: each building has its own action vector
        True   → centralised:  single concatenated action vector

    Returns
    -------
    env : CityLearnEnv
        Reset env (env.reset() has NOT been called yet).
    """
    env = CityLearnEnv(
        schema=str(SCHEMA_PATH),
        root_directory=str(DATA_DIR),
        central_agent=central_agent,
    )
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Building parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_building_params(building) -> dict:
    """Extract all physical parameters needed by the MPC for one building.

    Parameters
    ----------
    building : LSTMDynamicsBuilding
        A building object from `env.buildings`.

    Returns
    -------
    params : dict
        Keys:
          nominal_power_hp   (kW)   – heat-pump electrical nominal power
          nominal_power_bat  (kW)   – battery charge/discharge nominal power
          battery_capacity   (kWh)  – usable battery capacity (accounts for DOD)
          battery_efficiency (-)    – simplified round-trip efficiency (scalar)
          depth_of_discharge (-)    – fraction that may be discharged
          action_low         (1-D ndarray) – per-action lower bounds from env
          action_high        (1-D ndarray) – per-action upper bounds from env
          action_names       (list[str])   – active action names in order
    """
    hp  = building.cooling_device       # HeatPump
    bat = building.electrical_storage   # Battery

    # Simplified efficiency: average of the power–efficiency curve values
    # (the curve maps normalised power → efficiency; we take the mean)
    avg_eff = float(np.mean(bat.power_efficiency_curve[1]))

    params = {
        "nominal_power_hp":   float(hp.nominal_power),
        "nominal_power_bat":  float(bat.nominal_power),
        # Usable capacity already accounts for depth-of-discharge internally;
        # we expose the full capacity here and apply DOD as an SOC lower bound.
        "battery_capacity":   float(bat.capacity),
        "battery_efficiency": avg_eff,
        "depth_of_discharge": float(bat.depth_of_discharge),
        "action_low":         building.action_space.low.copy(),
        "action_high":        building.action_space.high.copy(),
        "action_names":       list(building.active_actions),
    }
    return params


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Exogenous data (forecast data known ahead of time from CSV)
# ─────────────────────────────────────────────────────────────────────────────

def load_exogenous_data(building, sim_start: int, sim_end: int) -> pd.DataFrame:
    """Return a DataFrame with the full time series of exogenous variables for
    one building over the simulation window [sim_start, sim_end] (inclusive).

    These values come from the pre-simulated CSV files and are NOT affected by
    the controller's actions.  They are used as known disturbance forecasts
    inside the MPC.

    Index : absolute time step (sim_start … sim_end)

    Columns
    -------
    outdoor_dry_bulb_temperature  [°C]
    direct_solar_irradiance       [W/m²]
    cooling_demand_baseline       [kWh thermal]  – demand from CSV (no control)
    dhw_demand                    [kWh thermal]
    non_shiftable_load            [kWh]
    solar_generation              [kWh]  – stored as NEGATIVE (reduces net load)
    cooling_setpoint              [°C]
    heating_setpoint              [°C]
    comfort_band                  [°C]
    hvac_mode                     (int)  – 1=cooling,2=heating,3=auto
    """
    es  = building.energy_simulation   # EnergySimulation
    wth = building.weather             # Weather

    # IMPORTANT: After env.reset(), all data arrays in energy_simulation and
    # weather are 0-indexed starting from the simulation start (step 0 = absolute
    # step sim_start).  Do NOT use absolute year indices (e.g. 3624) — they would
    # fall outside the array and return empty slices.
    # We use 0-based indices [0 : T] and attach absolute timesteps as the index.
    T = sim_end - sim_start + 1   # number of simulation steps (e.g. 720)

    df = pd.DataFrame({
        "outdoor_dry_bulb_temperature": wth.outdoor_dry_bulb_temperature[0:T],
        "direct_solar_irradiance":      wth.direct_solar_irradiance[0:T],
        # cooling_demand from CSV = what WOULD happen without any HVAC modulation.
        # After env.reset() and before apply_actions(), this is overwritten
        # by update_cooling_demand() at each step.  We keep the baseline here
        # for reference and for the uncontrollable net estimate.
        "cooling_demand_baseline":      es.cooling_demand[0:T],
        "dhw_demand":                   es.dhw_demand[0:T],
        "non_shiftable_load":           es.non_shiftable_load[0:T],
        # energy_simulation.solar_generation stores inverter_ac_power_per_kw
        # (raw PV output, W/kW).  Convert to kWh and negate: generation reduces
        # net consumption, so the contribution is negative by CityLearn convention.
        "solar_generation":             -building.pv.get_generation(
                                            es.solar_generation[0:T]
                                        ),
        "cooling_setpoint":             es.indoor_dry_bulb_temperature_cooling_set_point[0:T],
        "heating_setpoint":             es.indoor_dry_bulb_temperature_heating_set_point[0:T],
        "comfort_band":                 es.comfort_band[0:T],
        "hvac_mode":                    es.hvac_mode[0:T],
    }, index=range(sim_start, sim_end + 1))   # index = absolute timesteps

    return df


def load_district_target(sim_start: int, sim_end: int) -> np.ndarray:
    """Load the hourly portfolio-level load target [kWh] for the TX dataset.

    Parameters
    ----------
    sim_start, sim_end : int
        Absolute time-step indices (0-based, matching the CSV row number - 1).

    Returns
    -------
    target : np.ndarray, shape (sim_end - sim_start + 1,)
    """
    target_path = DATA_DIR / "district_target.csv"
    df = pd.read_csv(target_path)
    return df["district_load_target"].values[sim_start: sim_end + 1]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_cop(building, outdoor_temps: np.ndarray) -> np.ndarray:
    """Compute the cooling COP of a building's heat pump at given outdoor temps.

    Uses the Carnot-based COP formula embedded in CityLearn's HeatPump class:
        COP = efficiency * (T_cool_target + 273.15) / (T_outdoor - T_cool_target)
    clamped to [0, 20].

    Parameters
    ----------
    building     : LSTMDynamicsBuilding
    outdoor_temps: np.ndarray  – outdoor dry-bulb temperature [°C]

    Returns
    -------
    cop : np.ndarray – cooling COP (dimensionless)
    """
    return building.cooling_device.get_cop(np.asarray(outdoor_temps), heating=False)


def compute_uncontrollable_net(building, exog_df: pd.DataFrame) -> np.ndarray:
    """Compute the uncontrollable part of each building's net electricity
    consumption (i.e. what remains after removing battery and HVAC contributions).

    Formula:
        uncontrollable_net[k] = non_shiftable_load[k]
                                + dhw_demand[k] / dhw_efficiency
                                + solar_generation[k]     (negative value)

    Note: dhw action is INACTIVE in TX, so the DHW device always meets its full
    thermal demand → dhw_electricity = dhw_demand / heater_efficiency.

    Parameters
    ----------
    building : LSTMDynamicsBuilding
    exog_df  : pd.DataFrame  – output of load_exogenous_data()

    Returns
    -------
    uncontrollable_net : np.ndarray, same length as exog_df
    """
    dhw_eff    = float(building.dhw_device.efficiency)
    dhw_elec   = exog_df["dhw_demand"].values / dhw_eff
    solar      = exog_df["solar_generation"].values   # already negative

    return exog_df["non_shiftable_load"].values + dhw_elec + solar


def get_sim_window(env: CityLearnEnv) -> tuple[int, int]:
    """Return the (start, end) absolute time-step indices of the simulation."""
    tracker = env.episode_tracker
    return tracker.simulation_start_time_step, tracker.simulation_end_time_step
