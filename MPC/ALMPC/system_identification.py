"""
system_identification.py
------------------------
Option 4 System Identification: LSTM Linearisation via Automatic Differentiation.

Instead of collecting random-rollout data and fitting a black-box linear model,
we DIRECTLY differentiate through each building's pre-trained LSTM model to obtain
the local sensitivity (Jacobian) of the next indoor temperature with respect to:

  1. the current indoor temperature (auto-regressive coefficient  a)
  2. the current cooling demand    (control influence coefficient  b)

These are per-building, and are averaged over the whole simulation period to give
stable, representative coefficients.

Linearised temperature model (physical units):
    temp[k+1] ≈  a_i * temp[k]  +  b_i * cooling_demand[k]  +  d[k]

where d[k] captures all exogenous effects (weather, occupancy, solar, etc.)
and is estimated online by the MPC as an additive disturbance.

For MPC action variable u_cooling ∈ [action_low, action_high]:
    cooling_demand[k] = u_cooling[k] * nominal_power_hp * COP(T_outdoor[k])
so:
    ∂temp/∂u_cooling = b_i * nominal_power_hp * COP[k]   (time-varying gain)

Battery SOC is governed by an exact analytical equation (no identification needed):
    soc[k+1] = soc[k]  +  u_battery[k] * nominal_power_bat * efficiency / capacity
"""

from __future__ import annotations

import numpy as np
import torch

from utils import make_env, get_sim_window, load_exogenous_data

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_lstm_input_array(building) -> np.ndarray:
    """Replicate get_dynamics_input() from LSTMDynamicsBuilding as a plain
    numpy array so we can wrap it in a differentiable torch tensor.

    Returns
    -------
    x : np.ndarray, shape (lookback, n_inputs)
        Each column is one input variable (min-max normalised).
        The last row corresponds to the MOST RECENT time step for non-temp
        variables, and to (current - 1) for indoor_dry_bulb_temperature.
    """
    lstm = building.dynamics
    rows = []
    for i, name in enumerate(lstm.input_observation_names):
        if name == "indoor_dry_bulb_temperature":
            # Temperature: exclude the current (not-yet-predicted) slot → [:-1]
            rows.append(lstm._model_input[i][:-1])   # length = lookback
        else:
            # All other inputs: include the current slot → [1:]
            rows.append(lstm._model_input[i][1:])    # length = lookback
    # rows: list of (lookback,) arrays  →  stack → (n_inputs, lookback) → T
    return np.array(rows, dtype="float32").T         # (lookback, n_inputs)


def _compute_jacobians_single_step(building) -> tuple[float, float]:
    """Compute the linearisation coefficients at the CURRENT hidden state of
    the building's LSTM using PyTorch automatic differentiation.

    The LSTM maps (normalised input sequence) → (normalised temperature).
    We differentiate through this mapping to get:

        a_norm = ∂temp_norm[k] / ∂temp_norm[k-1]
                 (last temperature value in the lookback window)

        b_norm = ∂temp_norm[k] / ∂cooling_demand_norm[k]
                 (current-step cooling demand value in the lookback window)

    Then convert to physical units via the stored normalisation bounds.

    Returns
    -------
    a_phys : float
        Auto-regressive temperature coefficient (dimensionless, ∂T/∂T_prev).
    b_phys : float
        Cooling-demand sensitivity  [°C / kWh_thermal].
        Multiply by (nominal_power_hp * COP) to get ∂T/∂u_cooling.
    """
    lstm = building.dynamics

    # Index of each variable of interest
    idx_temp = lstm.input_observation_names.index("indoor_dry_bulb_temperature")
    idx_cd   = lstm.input_observation_names.index("cooling_demand")

    # Normalisation ranges
    temp_min  = lstm.input_normalization_minimum[idx_temp]
    temp_max  = lstm.input_normalization_maximum[idx_temp]
    cd_min    = lstm.input_normalization_minimum[idx_cd]
    cd_max    = lstm.input_normalization_maximum[idx_cd]
    temp_rng  = temp_max - temp_min   # > 0
    cd_rng    = cd_max   - cd_min     # > 0

    # Build input tensor with gradients enabled
    x_np   = _get_lstm_input_array(building)              # (lookback, n_inputs)
    x_leaf = torch.tensor(x_np).unsqueeze(0)              # (1, lookback, n_inputs)
    x_leaf = x_leaf.requires_grad_(True)

    # Use the CURRENT hidden state but detach it from any previous graph
    # (we only want gradients w.r.t. x, not through the hidden state history)
    h = tuple(s.detach().clone() for s in lstm._hidden_state)

    # Forward pass through the LSTM + linear head → normalised temperature
    with torch.enable_grad():
        out, _ = lstm(x_leaf, h)   # out: (1, 1)  normalised temp prediction
        out.backward()             # compute dout / dx_leaf

    grad = x_leaf.grad[0]          # (lookback, n_inputs)

    # ── Jacobian for indoor temperature (last time step in lookback window) ──
    # For indoor_dry_bulb_temperature the buffer uses [:-1], so index 12
    # (last in lookback) corresponds to k-1 (the previous measured temperature).
    a_norm = grad[-1, idx_temp].item()

    # ── Jacobian for cooling demand (last time step = current step k) ────────
    b_norm = grad[-1, idx_cd].item()

    # ── Convert to physical units ─────────────────────────────────────────────
    # ∂temp_phys / ∂temp_phys_prev = (temp_rng / temp_rng) * a_norm = a_norm
    a_phys = float(a_norm)

    # ∂temp_phys / ∂cd_phys  =  a_norm_cross * (temp_rng / cd_rng)
    b_phys = float(b_norm * temp_rng / cd_rng)

    return a_phys, b_phys


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def identify_building(
    building,
    exog_df,
    warmup_steps: int = 13,
) -> dict:
    """Compute the average LSTM linearisation coefficients for ONE building.

    Procedure
    ---------
    1. Warm up the LSTM hidden state for `warmup_steps` steps by feeding the
       true (uncontrolled) observations from the dataset — this puts the hidden
       state in a realistic operating region.
    2. For each subsequent step in the simulation window, compute the
       per-step Jacobians (a_k, b_k) via autograd.
    3. Return the mean and std of (a, b) across all steps.

    Because the LSTM hidden state evolves over time the Jacobians capture the
    ACTUAL operating trajectory of each building, not an artificial one.

    Parameters
    ----------
    building   : LSTMDynamicsBuilding  (already reset via env.reset())
    exog_df    : pd.DataFrame          from utils.load_exogenous_data()
    warmup_steps : int                 must be >= LSTM lookback (default 13)

    Returns
    -------
    result : dict with keys:
        a_mean, a_std  – auto-regressive temperature coefficient
        b_mean, b_std  – cooling-demand coefficient [°C / kWh_thermal]
        a_seq          – per-step a_k array
        b_seq          – per-step b_k array
    """
    n_steps = len(exog_df)
    assert warmup_steps >= building.dynamics.lookback, (
        f"warmup_steps ({warmup_steps}) must be >= LSTM lookback "
        f"({building.dynamics.lookback})"
    )

    a_seq: list[float] = []
    b_seq: list[float] = []

    for step_idx in range(n_steps):
        # Once we have enough history in the LSTM buffer, start recording
        if step_idx < warmup_steps:
            # During warm-up we just skip (the env handles LSTM state update
            # internally via apply_actions in each env.step() call)
            continue

        a_k, b_k = _compute_jacobians_single_step(building)
        a_seq.append(a_k)
        b_seq.append(b_k)

    a_arr = np.array(a_seq)
    b_arr = np.array(b_seq)

    return {
        "a_mean": float(np.mean(a_arr)),
        "a_std":  float(np.std(a_arr)),
        "b_mean": float(np.mean(b_arr)),
        "b_std":  float(np.std(b_arr)),
        "a_seq":  a_arr,
        "b_seq":  b_arr,
    }


def run_system_identification(save_path: str | None = None) -> dict:
    """Identify all 25 buildings in the TX dataset via LSTM linearisation.

    Runs one full episode with zero actions (so the LSTM evolves on the
    uncontrolled trajectory) while collecting per-step Jacobians.

    Parameters
    ----------
    save_path : str or None
        If given, saves the result as a .npz archive at this path.

    Returns
    -------
    sid_results : dict
        Keys are building names.  Each value is a dict with keys:
            a_mean, a_std  – averaged AR temperature coefficient
            b_mean, b_std  – averaged cooling-demand coefficient [°C/kWh]
        Plus compact arrays:
            all_a_mean, all_b_mean  – shape (n_buildings,) summary arrays
    """
    print("=" * 60)
    print("LSTM Linearisation – System Identification (Option 4)")
    print("=" * 60)

    # ── Build environment (decentralised so each building acts independently) ─
    env = make_env(central_agent=False)
    observations, _ = env.reset()

    sim_start, sim_end = get_sim_window(env)
    n_steps = sim_end - sim_start + 1
    n_buildings = len(env.buildings)

    print(f"Simulation window : steps {sim_start} – {sim_end}  ({n_steps} hours)")
    print(f"Number of buildings: {n_buildings}\n")

    # ── Load exogenous data for all buildings ──────────────────────────────────
    exog_list = [
        load_exogenous_data(b, sim_start, sim_end)
        for b in env.buildings
    ]

    # ── Zero-action episode: let LSTM warm up on natural trajectory ───────────
    # We collect per-step Jacobians inside the episode loop.
    # a_history[i][k], b_history[i][k] = Jacobians for building i at step k
    warmup = env.buildings[0].dynamics.lookback   # 13
    a_history = [[] for _ in range(n_buildings)]
    b_history = [[] for _ in range(n_buildings)]

    print("Running zero-action episode to warm up LSTM hidden states …")

    for step in range(n_steps):
        # Zero actions: [u_battery=0, u_cooling=0] per building
        # (0 = no battery charge, 0 = heat pump off)
        zero_actions = [[0.0, 0.0] for _ in range(n_buildings)]

        # ── Collect Jacobians BEFORE stepping (uses current hidden state) ──
        if step >= warmup:
            for i, building in enumerate(env.buildings):
                # Only record if LSTM is active (buffer is populated)
                if building.dynamics._model_input[0][0] is not None:
                    a_k, b_k = _compute_jacobians_single_step(building)
                    a_history[i].append(a_k)
                    b_history[i].append(b_k)

        # Step the environment (LSTM is updated internally here)
        observations, _, terminated, truncated, _ = env.step(zero_actions)

        if terminated or truncated:
            break

    # ── Aggregate results per building ────────────────────────────────────────
    sid_results: dict = {}
    all_a_mean = np.zeros(n_buildings)
    all_b_mean = np.zeros(n_buildings)

    print("\nBuilding-level identification results:")
    print(f"  {'Building':45s}  {'a_mean':>8s}  {'b_mean':>10s}  {'b_std':>10s}")
    print("  " + "-" * 80)

    for i, building in enumerate(env.buildings):
        name   = building.name
        a_arr  = np.array(a_history[i])
        b_arr  = np.array(b_history[i])

        result = {
            "a_mean": float(np.mean(a_arr)),
            "a_std":  float(np.std(a_arr)),
            "b_mean": float(np.mean(b_arr)),
            "b_std":  float(np.std(b_arr)),
            "a_seq":  a_arr,
            "b_seq":  b_arr,
        }
        sid_results[name]  = result
        all_a_mean[i]      = result["a_mean"]
        all_b_mean[i]      = result["b_mean"]

        print(f"  {name:45s}  {result['a_mean']:8.4f}  "
              f"{result['b_mean']:10.6f}  {result['b_std']:10.6f}")

    sid_results["_summary"] = {
        "building_names": [b.name for b in env.buildings],
        "all_a_mean":     all_a_mean,
        "all_b_mean":     all_b_mean,
    }

    # ── Optional save ─────────────────────────────────────────────────────────
    if save_path is not None:
        np.savez(
            save_path,
            building_names=np.array([b.name for b in env.buildings]),
            all_a_mean=all_a_mean,
            all_b_mean=all_b_mean,
            **{f"a_seq_{i}": v["a_seq"] for i, (_, v) in
               enumerate(sid_results.items()) if _ != "_summary"},
            **{f"b_seq_{i}": v["b_seq"] for i, (_, v) in
               enumerate(sid_results.items()) if _ != "_summary"},
        )
        print(f"\nSaved SID results → {save_path}")

    print("\nSystem identification complete.")
    return sid_results


# ─────────────────────────────────────────────────────────────────────────────
# Online Jacobian helper (used by the MPC at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def get_online_jacobians(building) -> tuple[float, float]:
    """Compute LSTM Jacobians at the CURRENT operating point of a building.

    This is called by the MPC controller at every time step so that the
    linearisation tracks the actual operating trajectory (online re-linearisation).

    Returns
    -------
    a_phys : float   auto-regressive temperature coefficient
    b_phys : float   cooling-demand sensitivity [°C / kWh_thermal]
    """
    # Guard: if LSTM buffer is not yet populated, return neutral defaults
    if building.dynamics._model_input[0][0] is None:
        return 1.0, 0.0

    return _compute_jacobians_single_step(building)


if __name__ == "__main__":
    # Run SID standalone and save results
    save_to = Path(__file__).parent / "saved_files" / "lstm_jacobians.npz"
    save_to.parent.mkdir(parents=True, exist_ok=True)
    run_system_identification(save_path=str(save_to))
