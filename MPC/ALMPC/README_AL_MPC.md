# Adaptive LSTM-Linearized MPC (AL-MPC)

## Technical README for the CityLearn Texas District Load Control Challenge

**Original implementation name:** `MPC_Decentralized_1`  

**Recommended public/report name:** **Adaptive LSTM-Linearized MPC (AL-MPC)**  
**Control architecture:** Decentralized Model Predictive Control  
**Environment:** CityLearn / IEA EBC Annex 96 Common Exercise 1 / Texas climate dataset  
**Buildings:** 25 residential buildings  
**Simulation period:** June, 720 hourly timesteps  
**Primary task:** Track a district-level target load profile while maintaining indoor thermal comfort.

---

## 1. Why rename `MPC_Decentralized_1`?

The internal name `MPC_Decentralized_1` is useful during development, but it is not descriptive enough for a public report or repository. The controller is more accurately described as:

> **Adaptive LSTM-Linearized MPC (AL-MPC)**

This name reflects the actual methodology:

1. **Adaptive**: the controller updates its local linear model at every timestep using the current operating point of the building.
2. **LSTM-Linearized**: the building thermal model is obtained by differentiating through CityLearn's pre-trained LSTM dynamics model.
3. **MPC**: the controller solves a finite-horizon constrained optimization problem and applies only the first action of the optimal sequence.

The controller is not a static linear MPC and not a black-box learning controller. It is a hybrid method that combines a learned nonlinear dynamics model with classical convex MPC.

---

## 2. High-level method summary

AL-MPC is a decentralized MPC controller for the CityLearn Texas neighborhood. Each building is controlled by its own independent MPC instance. The controller at each building uses local information only, solves its own optimization problem, and sends a two-dimensional action to CityLearn:

$$
u_i(k) = [u_{\mathrm{bat},i}(k), u_{\mathrm{cool},i}(k)]
$$

where:

- $u_{\mathrm{bat},i}$ controls battery charging/discharging,
- $u_{\mathrm{cool},i}$ controls the cooling device power level.

The controller has three main parts:

1. **System identification / local model extraction**  
   Instead of fitting a separate linear model from random rollout data, AL-MPC differentiates through the existing CityLearn LSTM building dynamics model and extracts local Jacobians.

2. **Per-building target decomposition**  
   The district target is split across buildings proportionally to each building's uncontrollable baseline net load.

3. **Convex MPC optimization**  
   At every timestep, each building solves a finite-horizon quadratic optimization problem with tracking, comfort, smoothness, net-load smoothness, and terminal battery state-of-charge costs.

---

## 3. File structure

The controller is implemented across the following files.

```text
MPC_Decentralised_1/
│
├── mpc_agent.py
├── mpc_controller.py
├── run_mpc.py
├── system_identification.py
├── utils.py
└── mpc_submission.ipynb
```

### 3.1 `mpc_agent.py`

Provides the CityLearn-compatible `MPCAgent` wrapper. Its role is to expose a `.predict(observations)` method, similar to CityLearn's built-in agents. This allows the MPC controller to be used in the standard CityLearn simulation loop:

```python
agent = MPCAgent(env)
while not env.terminated:
    actions = agent.predict(obs)
    obs, _, _, _, _ = env.step(actions)
```

The environment must be initialized with:

```python
env = make_env(central_agent=False)
```

because AL-MPC is decentralized: one MPC controller is created per building.

### 3.2 `mpc_controller.py`

Contains the main `BuildingMPC` class. This is the core optimization controller. Each instance controls one building and solves a CVXPY optimization problem at every timestep.

### 3.3 `system_identification.py`

Implements LSTM linearization through PyTorch automatic differentiation. It extracts local sensitivity coefficients of the next indoor temperature with respect to:

- previous indoor temperature,
- current cooling demand.

This file contains both offline analysis utilities and the online Jacobian function used at runtime.

### 3.4 `run_mpc.py`

Standalone script for running AL-MPC over the full Texas simulation window. It creates the environment, loads exogenous data, splits the district target, initializes one MPC per building, runs the 720-hour episode, computes KPIs, plots results, and optionally saves outputs.

### 3.5 `utils.py`

Shared helper utilities for:

- constructing the CityLearn environment,
- extracting physical device parameters,
- loading exogenous variables,
- computing heat pump COP,
- computing uncontrollable net load,
- reading the simulation window.

### 3.6 `mpc_submission.ipynb`

Notebook used for running the controller and logging the final results to Weights & Biases. It uses the same `MPCAgent` interface and logs district load, KPIs, and building temperature plots.

---

## 4. Control architecture

AL-MPC uses a decentralized architecture.

Let there be $N = 25$ buildings. Instead of solving one large centralized MPC problem for the whole district, the controller creates:

$$
N = 25
$$

independent MPC controllers:

$$
\mathcal{C}_1, \mathcal{C}_2, \ldots, \mathcal{C}_{25}
$$

Each controller solves:

$$
\min_{u_i(0), \ldots, u_i(H-1)} J_i
$$

using only building-specific variables and forecasts.

The advantage of this design is scalability. Each optimization problem remains low-dimensional, because every controller only optimizes two control actions per timestep:

$$
u_i(k) = [u_{\mathrm{bat},i}(k), u_{\mathrm{cool},i}(k)].
$$

The trade-off is that there is no explicit communication between buildings. District-level coordination is introduced indirectly through the per-building target split.

---

## 5. State, action, and disturbance variables

### 5.1 State vector

For each building $i$, the MPC state is:

$$
x_i(k) =
\begin{bmatrix}
T_i(k) \\
\mathrm{SOC}_i(k)
\end{bmatrix}
$$

where:

- $T_i(k)$ is the indoor dry-bulb temperature in degrees Celsius,
- $\mathrm{SOC}_i(k)$ is the electrical storage state of charge.

### 5.2 Action vector

For each building, the controller chooses:

$$
u_i(k) =
\begin{bmatrix}
u_{\mathrm{bat},i}(k) \\
u_{\mathrm{cool},i}(k)
\end{bmatrix}
$$

where:

- $u_{\mathrm{bat},i}(k) < 0$: battery discharging,
- $u_{\mathrm{bat},i}(k) > 0$: battery charging,
- $u_{\mathrm{cool},i}(k) = 0$: cooling device off,
- $u_{\mathrm{cool},i}(k) = 1$: cooling device at full available power.

Action bounds are not manually hard-coded. They are extracted from each building's CityLearn action space:

```python
self.u_low  = params["action_low"]
self.u_high = params["action_high"]
```

### 5.3 Exogenous variables

The MPC uses known exogenous inputs over the horizon, including:

- outdoor dry-bulb temperature,
- non-shiftable load,
- domestic hot water demand converted to electrical demand,
- solar generation,
- uncontrollable net load,
- district target allocation.

These are treated as forecasts available to the controller.

---

## 6. System identification method

### 6.1 Motivation

Classical MPC requires an explicit model of building dynamics. A common approach is to fit a linear model from input-output data, but this can be inaccurate and requires exploratory rollouts. AL-MPC avoids this by using the building dynamics already embedded in CityLearn.

CityLearn uses an LSTM-based building dynamics model for indoor temperature prediction. AL-MPC treats this LSTM as a differentiable nonlinear model and extracts a local linear approximation at the current operating point.

### 6.2 Nonlinear LSTM dynamics

The true CityLearn thermal model can be represented abstractly as:

$$
T_i(k+1) = f_i(z_i(k))
$$

where $z_i(k)$ is the LSTM input sequence containing past and current normalized observations, including indoor temperature, cooling demand, weather variables, and other building-related inputs.

The function $f_i(\cdot)$ is nonlinear and time-varying because it depends on the internal LSTM hidden state.

### 6.3 Local linearization

At each timestep, the LSTM is locally linearized with respect to two variables:

1. previous indoor temperature,
2. current cooling demand.

The extracted Jacobians are:

$$
a_i(k) = \frac{\partial T_i(k+1)}{\partial T_i(k)}
$$

and

$$
b_i(k) = \frac{\partial T_i(k+1)}{\partial q_{\mathrm{cool},i}(k)}
$$

where $q_{\mathrm{cool},i}(k)$ is the thermal cooling demand.

The local linear temperature model is:

$$
T_i(k+1) \approx a_i(k)T_i(k) + b_i(k)q_{\mathrm{cool},i}(k) + d_i(k)
$$

where $d_i(k)$ is an additive disturbance term capturing effects not directly modeled by the two-variable linearization, such as weather, occupancy, solar effects, and internal gains.

### 6.4 Automatic differentiation through the LSTM

The function `_compute_jacobians_single_step(building)` performs the following steps:

1. Reconstructs the normalized LSTM input sequence used by the CityLearn building model.
2. Wraps it as a PyTorch tensor with `requires_grad=True`.
3. Runs a forward pass through the LSTM using the current hidden state.
4. Calls `.backward()` on the predicted normalized temperature.
5. Reads gradients with respect to:
   - the last indoor temperature input,
   - the current cooling demand input.
6. Converts gradients from normalized units into physical units.

The normalized gradient for temperature is directly dimensionless:

$$
a_{\mathrm{phys}} = a_{\mathrm{norm}}
$$

because both input and output are temperatures normalized by the same range.

For cooling demand, the gradient must be converted from normalized units to physical units:

$$
b_{\mathrm{phys}}
=
b_{\mathrm{norm}}
\frac{T_{\max} - T_{\min}}{q_{\max} - q_{\min}}
$$

where:

- $T_{\max} - T_{\min}$ is the physical temperature normalization range,
- $q_{\max} - q_{\min}$ is the physical cooling demand normalization range.

The resulting coefficient $b_{\mathrm{phys}}$ has units:

$$
^\circ\mathrm{C}/\mathrm{kWh}_{\mathrm{thermal}}
$$

### 6.5 Offline system identification utility

The file `system_identification.py` also provides a standalone function:

```python
run_system_identification(save_path=None)
```

This routine performs a zero-action rollout through the Texas environment and collects Jacobians after the LSTM lookback buffer is populated. The warmup length is:

$$
\text{warmup} = 13
$$

matching the LSTM lookback length.

For each building, it records:

$$
a_i(k), \quad b_i(k)
$$

across the simulation window and reports:

$$
\bar{a}_i, \quad \sigma_{a_i}, \quad \bar{b}_i, \quad \sigma_{b_i}
$$

This offline SID step is mainly diagnostic. The deployed controller does not rely on fixed average coefficients. Instead, it recomputes the Jacobians online at every timestep.

### 6.6 Online re-linearization

At runtime, the MPC calls:

```python
get_online_jacobians(building)
```

every time it solves the optimization problem. Therefore, the model coefficients are not fixed constants; they change with the current LSTM state and operating condition.

This gives the controller a successive-linearization structure:

$$
(a_i(k), b_i(k)) = \nabla f_i(z_i(k))
$$

The linear MPC problem is convex at each timestep, but the model is updated repeatedly as the nonlinear building dynamics evolve.

---

## 7. Temperature model used inside MPC

The MPC uses the local linear model:

$$
T_i(k+j+1)
=
a_i(k)T_i(k+j)
+
B_i(k+j)u_{\mathrm{cool},i}(k+j)
+
d_i(k)
$$

where:

$$
B_i(k+j)
=
b_i(k)P_{\mathrm{hp},i}\mathrm{COP}_i(k+j)
$$

and:

- $P_{\mathrm{hp},i}$ is the nominal heat-pump electrical power,
- $\mathrm{COP}_i(k+j)$ is the forecast cooling coefficient of performance,
- $u_{\mathrm{cool},i}(k+j)$ is the cooling control action.

The reason for multiplying by COP is that the LSTM sensitivity $b_i$ is with respect to thermal cooling demand, but the MPC action controls normalized electrical cooling power. The conversion is:

$$
q_{\mathrm{cool},i}(k)
=
u_{\mathrm{cool},i}(k)
P_{\mathrm{hp},i}
\mathrm{COP}_i(k)
$$

Thus:

$$
\frac{\partial T_i(k+1)}{\partial u_{\mathrm{cool},i}(k)}
=
b_i(k)P_{\mathrm{hp},i}\mathrm{COP}_i(k)
$$

Because cooling reduces indoor temperature, $b_i$ is typically negative.

---

## 8. Heat pump COP model

The controller uses CityLearn's heat pump COP calculation through:

```python
building.cooling_device.get_cop(outdoor_temps, heating=False)
```

The helper function `compute_cop` wraps this call. The COP depends on outdoor temperature and is computed using the heat-pump model embedded in CityLearn. Conceptually:

$$
\mathrm{COP}(k) = g(T_{\mathrm{out}}(k))
$$

where $T_{\mathrm{out}}$ is the outdoor dry-bulb temperature.

The COP forecast over the MPC horizon is used to create a time-varying cooling gain:

$$
B_i(k), B_i(k+1), \ldots, B_i(k+H-1)
$$

This means that the same cooling action can have different thermal effects depending on forecast outdoor temperature.

---

## 9. Disturbance estimation

The LSTM linearization only includes indoor temperature and cooling demand explicitly. All other effects are grouped into an additive disturbance term.

At each timestep, the disturbance is updated using feedback from the previous observed transition:

$$
\hat{d}_i(k)
=
T_i(k)
-
a_i(k)T_i(k-1)
-
b_i(k)q_{\mathrm{cool},i}(k-1)
$$

In code:

```python
self.d_temp = temp_obs - a_phys * prev_temp - b_phys * prev_cooling_demand
```

This term is then held constant over the MPC horizon:

$$
d_i(k+j) \approx \hat{d}_i(k), \quad j=0,\ldots,H-1
$$

This feedback correction is important because it allows the MPC to compensate for exogenous effects and model mismatch without explicitly modeling all disturbances.

---

## 10. Battery model

The battery state-of-charge dynamics are modeled analytically and do not require system identification.

For each building:

$$
\mathrm{SOC}_i(k+1)
=
\mathrm{SOC}_i(k)
+
\alpha_{\mathrm{bat},i}u_{\mathrm{bat},i}(k)
$$

where:

$$
\alpha_{\mathrm{bat},i}
=
\frac{P_{\mathrm{bat},i}\eta_i}{C_{\mathrm{bat},i}}
$$

with:

- $P_{\mathrm{bat},i}$: battery nominal power,
- $\eta_i$: average battery efficiency,
- $C_{\mathrm{bat},i}$: battery capacity.

The implementation computes:

```python
self.alpha_bat = self.P_bat * self.eta / self.C_bat
```

The battery efficiency is approximated as the mean of the CityLearn battery power-efficiency curve:

```python
avg_eff = float(np.mean(bat.power_efficiency_curve[1]))
```

The SOC is constrained between an effective lower bound and full charge:

$$
\mathrm{SOC}_{\min,i} \leq \mathrm{SOC}_i(k+j) \leq 1
$$

The nominal lower bound is:

$$
\mathrm{SOC}_{\min,i}=1-\mathrm{DOD}_i
$$

where $\mathrm{DOD}_i$ is the depth of discharge.

To avoid infeasibility when the observed SOC is already below this bound, the controller uses:

$$
\mathrm{SOC}_{\min,i}^{\mathrm{eff}}
=
\min(\mathrm{SOC}_{\min,i}, \mathrm{SOC}_i(k))
$$

This ensures the optimization remains feasible and allows the controller to charge the battery back toward the normal operating range.

---

## 11. Uncontrollable net load model

The controller separates each building's total net electricity consumption into controllable and uncontrollable components.

The uncontrollable component is computed as:

$$
L_{\mathrm{unc},i}(k)
=
L_{\mathrm{nonshift},i}(k)
+
\frac{D_{\mathrm{dhw},i}(k)}{\eta_{\mathrm{dhw},i}}
+
G_{\mathrm{solar},i}(k)
$$

where:

- $L_{\mathrm{nonshift},i}$ is non-shiftable electrical load,
- $D_{\mathrm{dhw},i}$ is domestic hot water thermal demand,
- $\eta_{\mathrm{dhw},i}$ is DHW heater efficiency,
- $G_{\mathrm{solar},i}$ is solar generation, stored as a negative contribution.

In code:

```python
uncontrollable_net = non_shiftable_load + dhw_demand / dhw_efficiency + solar_generation
```

The controllable component is:

$$
L_{\mathrm{ctrl},i}(k)
=
P_{\mathrm{hp},i}u_{\mathrm{cool},i}(k)
+
P_{\mathrm{bat},i}u_{\mathrm{bat},i}(k)
$$

Therefore, total building net load is:

$$
L_i(k)
=
L_{\mathrm{unc},i}(k)
+
P_{\mathrm{hp},i}u_{\mathrm{cool},i}(k)
+
P_{\mathrm{bat},i}u_{\mathrm{bat},i}(k)
$$

---

## 12. District target decomposition

The official target is given at district level:

$$
D^*(k)
$$

However, AL-MPC is decentralized, so each building needs its own target. The implementation divides the district target among buildings proportionally to each building's uncontrollable net load.

For building $i$, the weight is:

$$
w_i(k)
=
\frac{L_{\mathrm{unc},i}(k)}{\sum_{m=1}^{N}L_{\mathrm{unc},m}(k)}
$$

The per-building target is:

$$
D_i^*(k)
=
w_i(k)D^*(k)
$$

If the total uncontrollable load is zero, the implementation falls back to equal weights:

$$
w_i(k)=\frac{1}{N}
$$

This target allocation is implemented in:

```python
split_target_proportional(district_target, uncontrollable)
```

This design gives larger buildings a larger share of the district target and avoids forcing all buildings to follow identical target values.

---

## 13. MPC prediction horizon

The controller is parameterized by a prediction horizon:

$$
H
$$

The standalone script default is:

```python
horizon = 24
```

The submission notebook used:

```python
horizon = 24
```

The `run_mpc.py` `__main__` example also contains a separate experiment configuration with:

```python
horizon = 36
```

For the W&B submission, the relevant hyperparameter configuration was:

```python
agent_mpc = MPCAgent(
    env,
    horizon        = 24,
    w_track        = 50.0,
    w_comfort      = 20.0,
    w_smooth       = 1.0,
    w_net_smooth   = 50.0,
    w_terminal_soc = 5.0,
)
```

Thus, the submitted AL-MPC run used a **24-hour prediction horizon**.

---

## 14. MPC optimization problem

At every timestep and for each building, AL-MPC solves a finite-horizon quadratic program.

### 14.1 Decision variables

For a horizon $H$, the optimization variables are:

$$
u(0), \ldots, u(H-1)
$$

where each action has two components:

$$
u(j) =
\begin{bmatrix}
u_{\mathrm{bat}}(j) \\
u_{\mathrm{cool}}(j)
\end{bmatrix}
$$

The optimization also includes predicted trajectories:

$$
T(0), \ldots, T(H)
$$

$$
\mathrm{SOC}(0), \ldots, \mathrm{SOC}(H)
$$

and nonnegative comfort slack variables:

$$
s_{\mathrm{hot}}(j) \geq 0,
\quad
s_{\mathrm{cold}}(j) \geq 0
$$

### 14.2 Initial conditions

The first predicted state is constrained to the current observed state:

$$
T(0)=T_{\mathrm{obs}}
$$

$$
\mathrm{SOC}(0)=\mathrm{SOC}_{\mathrm{obs}}
$$

### 14.3 Dynamics constraints

For each prediction step $j=0,\ldots,H-1$:

$$
T(j+1)=aT(j)+B(j)u_{\mathrm{cool}}(j)+d
$$

$$
\mathrm{SOC}(j+1)=\mathrm{SOC}(j)+\alpha_{\mathrm{bat}}u_{\mathrm{bat}}(j)
$$

### 14.4 Action constraints

The controller enforces CityLearn action bounds:

$$
u_{\min} \leq u(j) \leq u_{\max}
$$

or componentwise:

$$
u_{\mathrm{bat}}^{\min}
\leq
u_{\mathrm{bat}}(j)
\leq
u_{\mathrm{bat}}^{\max}
$$

$$
u_{\mathrm{cool}}^{\min}
\leq
u_{\mathrm{cool}}(j)
\leq
u_{\mathrm{cool}}^{\max}
$$

### 14.5 SOC constraints

$$
\mathrm{SOC}_{\min}^{\mathrm{eff}}
\leq
\mathrm{SOC}(j+1)
\leq
1
$$

### 14.6 Soft comfort constraints

The Texas comfort range is fixed at:

$$
22^\circ\mathrm{C} \leq T \leq 26^\circ\mathrm{C}
$$

The controller uses:

$$
T_{\mathrm{set}} = 24^\circ\mathrm{C}
$$

$$
\Delta T = 2^\circ\mathrm{C}
$$

so:

$$
T_{\min}=T_{\mathrm{set}}-\Delta T=22^\circ\mathrm{C}
$$

$$
T_{\max}=T_{\mathrm{set}}+\Delta T=26^\circ\mathrm{C}
$$

Comfort is implemented as a soft constraint:

$$
T(j+1) \geq T_{\min} - s_{\mathrm{cold}}(j)
$$

$$
T(j+1) \leq T_{\max} + s_{\mathrm{hot}}(j)
$$

with:

$$
s_{\mathrm{hot}}(j), s_{\mathrm{cold}}(j) \geq 0
$$

This avoids infeasibility if the building cannot physically maintain the comfort range.

---

## 15. MPC cost function

The total cost minimized by each building controller is:

$$
J = J_{\mathrm{track}}
+J_{\mathrm{comfort}}
+J_{\mathrm{action}}
+J_{\mathrm{net}}
+J_{\mathrm{terminal}}
$$

Each term is described below.

### 15.1 Load tracking cost

The primary objective is to make the building net load follow its allocated target:

$$
J_{\mathrm{track}}
=
\sum_{j=0}^{H-1}
w_{\mathrm{track}}
\left(L(j)-D^*(j)\right)^2
$$

where:

- $L(j)$ is predicted building net load,
- $D^*(j)$ is the building's allocated target,
- $w_{\mathrm{track}}$ is the tracking weight.

### 15.2 Comfort slack cost

Temperature violations are penalized through slack variables:

$$
J_{\mathrm{comfort}}
=
\sum_{j=0}^{H-1}
w_{\mathrm{comfort}}
\left(s_{\mathrm{hot}}(j)^2+s_{\mathrm{cold}}(j)^2\right)
$$

This discourages comfort violations but does not make comfort constraints infeasible.

### 15.3 Action smoothness cost

The controller penalizes abrupt changes in actions:

$$
J_{\mathrm{action}}
=
\sum_{j=0}^{H-1}
w_{\mathrm{smooth}}
\left\|u(j)-u(j-1)\right\|_2^2
$$

For $j=0$, the previous action is the action applied in the previous real timestep.

This term discourages rapid switching in battery and cooling commands.

### 15.4 Net-load smoothness cost

A separate smoothness term penalizes step-to-step changes in predicted net load:

$$
J_{\mathrm{net}}
=
\sum_{j=0}^{H-1}
w_{\mathrm{net}}
\left(L(j)-L(j-1)\right)^2
$$

For $j=0$, the previous net load is the actual net load from the previous real timestep, stored as `prev_net_total`.

This term directly reduces noisy hour-to-hour oscillations in the produced district load.

### 15.5 Terminal SOC cost

The controller penalizes ending the horizon with the battery too full or too empty:

$$
J_{\mathrm{terminal}}
=
w_{\mathrm{terminal}}
\left(\mathrm{SOC}(H)-0.5\right)^2
$$

This term was included to reduce finite-horizon battery cycling. Without it, the controller may repeatedly drain or fill the battery near the end of each horizon window.

---

## 16. Complete optimization problem

For each building at each timestep, AL-MPC solves:

$$
\begin{aligned}
\min_{u,T,\mathrm{SOC},s_{\mathrm{hot}},s_{\mathrm{cold}}}
\quad &
\sum_{j=0}^{H-1}
\Big[
    w_{\mathrm{track}}(L(j)-D^*(j))^2 \\
&\quad + w_{\mathrm{comfort}}(s_{\mathrm{hot}}(j)^2+s_{\mathrm{cold}}(j)^2) \\
&\quad + w_{\mathrm{smooth}}\|u(j)-u(j-1)\|_2^2 \\
&\quad + w_{\mathrm{net}}(L(j)-L(j-1))^2
\Big] \\
&\quad + w_{\mathrm{terminal}}(\mathrm{SOC}(H)-0.5)^2
\end{aligned}
$$

subject to:

$$
T(0)=T_{\mathrm{obs}},
\quad
\mathrm{SOC}(0)=\mathrm{SOC}_{\mathrm{obs}}
$$

$$
T(j+1)=aT(j)+B(j)u_{\mathrm{cool}}(j)+d
$$

$$
\mathrm{SOC}(j+1)=\mathrm{SOC}(j)+\alpha_{\mathrm{bat}}u_{\mathrm{bat}}(j)
$$

$$
u_{\min}\leq u(j)\leq u_{\max}
$$

$$
\mathrm{SOC}_{\min}^{\mathrm{eff}} \leq \mathrm{SOC}(j+1) \leq 1
$$

$$
T(j+1) \geq 22 - s_{\mathrm{cold}}(j)
$$

$$
T(j+1) \leq 26 + s_{\mathrm{hot}}(j)
$$

$$
s_{\mathrm{hot}}(j),s_{\mathrm{cold}}(j)\geq 0
$$

This is a quadratic program because:

- the objective is quadratic,
- the constraints are linear,
- the linearization coefficients are fixed during each solve.

---

## 17. Solver implementation

The optimization is implemented using CVXPY.

Primary solver:

```python
problem.solve(solver=cp.OSQP, warm_starting=True, verbose=False)
```

Fallback solver:

```python
problem.solve(solver=cp.SCS, verbose=False)
```

The solver workflow is:

1. Build CVXPY variables.
2. Add initial-state constraints.
3. Add linearized temperature dynamics.
4. Add battery SOC dynamics.
5. Add action and SOC bounds.
6. Add soft comfort constraints.
7. Accumulate quadratic cost terms.
8. Solve using OSQP.
9. If OSQP fails, retry using SCS.
10. Apply only the first optimal action.

The applied action is clipped to the hard action bounds for numerical safety:

```python
u_opt[0] = np.clip(u_opt[0], self.u_low[0], self.u_high[0])
u_opt[1] = np.clip(u_opt[1], self.u_low[1], self.u_high[1])
```

If the problem is not solved successfully, the controller repeats the previous action clipped to the action bounds.

---

## 18. Receding-horizon control loop

The controller follows a standard receding-horizon MPC structure:

1. Observe current indoor temperature and battery SOC.
2. Extract online LSTM Jacobians.
3. Build forecasts for outdoor temperature, uncontrollable load, and target allocation.
4. Estimate additive disturbance from the previous step.
5. Solve the horizon optimization problem.
6. Apply only the first action.
7. Step the CityLearn environment.
8. Store action, temperature, SOC, and net load for the next iteration.
9. Repeat until the episode terminates.

In pseudocode:

```text
for each timestep k:
    for each building i:
        observe T_i(k), SOC_i(k)
        compute a_i(k), b_i(k) using LSTM autograd
        compute COP forecast
        estimate disturbance d_i(k)
        build H-step MPC problem
        solve quadratic program
        apply first action [u_bat_i(k), u_cool_i(k)]
    step CityLearn environment
```

---

## 19. Hyperparameters

### 19.1 Controller defaults in `MPCAgent`

The `MPCAgent` constructor uses:

```python
horizon        = 24
w_track        = 10.0
w_comfort      = 5.0
w_smooth       = 1.0
w_net_smooth   = 10.0
w_terminal_soc = 5.0
```

### 19.2 Submission notebook configuration

The W&B submission notebook used:

```python
horizon        = 24
w_track        = 50.0
w_comfort      = 20.0
w_smooth       = 1.0
w_net_smooth   = 50.0
w_terminal_soc = 5.0
```

### 19.3 Standalone `run_mpc.py` experiment configuration

The `__main__` experiment in `run_mpc.py` used:

```python
horizon        = 36
w_track        = 100.0
w_comfort      = 5.0
w_smooth       = 100.0
w_net_smooth   = 100.0
w_terminal_soc = 5.0
```

### 19.4 Interpretation of weights

| Hyperparameter | Meaning |
|---|---|
| `horizon` | Number of future hours optimized at each timestep. |
| `w_track` | Weight on building-level target tracking error. |
| `w_comfort` | Weight on violation of the 22--26 °C comfort band. |
| `w_smooth` | Weight on changes in battery/cooling actions. |
| `w_net_smooth` | Weight on changes in predicted net load. |
| `w_terminal_soc` | Weight on ending the horizon near SOC = 0.5. |

---

## 20. WandB experiment logging

The submission notebook logs the AL-MPC run to the shared Weights & Biases project:

```python
WANDB_ENTITY = "CityLearn-TeamB"
WANDB_PROJECT = "CityLearn"
```

The run was initialized with:

```python
name = "MPC_Decentralised_1"
```

For public documentation, this can be renamed to:

```python
name = "Adaptive LSTM-Linearized MPC"
```

or:

```python
name = "AL-MPC"
```

The notebook logs:

1. Hourly district load during simulation.
2. Final KPI values:
   - NMBE,
   - CV-RMSE,
   - temperature comfort violation.
3. Interactive Plotly figure for indoor temperatures of all 25 buildings.
4. Mean indoor temperature trajectory.
5. Comfort band between 22 °C and 26 °C.

The district load is logged during the simulation loop, while KPIs and temperature plots are logged after the full trajectory has been collected.

---

## 21. KPI calculation

The controller is evaluated using the same KPI logic as the rest of the project.

### 21.1 NMBE

$$
\mathrm{NMBE}
=
\frac{\mathrm{mean}(y-y_{\mathrm{ref}})}{\mathrm{mean}(y_{\mathrm{ref}})}\times 100
$$

where:

- $y$ is the district net load,
- $y_{\mathrm{ref}}$ is the target load.

### 21.2 CV-RMSE

$$
\mathrm{CV\text{-}RMSE}
=
\frac{\sqrt{\mathrm{mean}((y-y_{\mathrm{ref}})^2)}}{\mathrm{mean}(y_{\mathrm{ref}})}\times 100
$$

### 21.3 Temperature comfort violation

$$
\mathrm{Violation}
=
\mathrm{mean}(T<22 \;\mathrm{or}\; T>26)\times 100
$$

The comfort violation is calculated across all building-temperature samples.

---

## 22. Implementation details that matter

### 22.1 State is read from building objects, not raw observation arrays

Although `predict(observations)` accepts the current observation list, the controller reads indoor temperature and SOC directly from CityLearn building objects:

```python
b.observations()["indoor_dry_bulb_temperature"]
b.observations()["electrical_storage_soc"]
```

This reduces indexing errors and ensures precise access to named variables.

### 22.2 Forecast indices are clamped near the episode end

Near the end of the simulation, the forecast horizon may extend beyond the available dataset. The implementation clamps the forecast index to the final valid timestep:

```python
min(abs_step + j, self._sim_start + self._T - 1)
```

This prevents index errors and keeps forecast arrays length $H$.

### 22.3 Initial net-load memory avoids artificial first-step spikes

Each controller initializes `prev_net_total` to the first-step uncontrollable load:

```python
ctrl.prev_net_total = float(unctrllable[0, i])
```

This avoids a large artificial penalty at the first step due to comparing the first predicted net load against zero.

### 22.4 Actual simulated length is trimmed

The standalone script trims all arrays to the number of steps actually completed. This avoids zero-padded tails causing fake KPI or plotting artifacts.

---

## 23. Strengths of AL-MPC

### 23.1 Uses existing CityLearn dynamics model

The controller does not require training a separate building model. It uses the already available LSTM dynamics model embedded in the environment.

### 23.2 Online adaptation

The local model is recomputed at every timestep, allowing the MPC to adapt to changing operating conditions.

### 23.3 Convex optimization at each step

Although the underlying LSTM is nonlinear, each MPC optimization problem is convex because the LSTM is linearized before the solve.

### 23.4 Interpretable objective function

Each term in the cost function has a clear purpose: tracking, comfort, action smoothness, net-load smoothness, and terminal SOC management.

### 23.5 Scalable decentralized structure

Each building solves a small optimization problem independently. This avoids the large action and state spaces associated with centralized district-level MPC.

---

## 24. Limitations of AL-MPC

### 24.1 No explicit building-to-building coordination

The controller is decentralized. Coordination only occurs indirectly through target splitting. Buildings do not communicate during optimization.

### 24.2 Linearization validity is local

The LSTM linear model is only accurate near the current operating point. Large action changes may reduce prediction accuracy.

### 24.3 Disturbance is held constant across the horizon

The disturbance estimate captures unmodeled effects, but it is assumed constant over the prediction horizon.

### 24.4 Computational cost

Each building solves a CVXPY optimization problem at every timestep. With 25 buildings and 720 hours, this requires many solver calls:

$$
25 \times 720 = 18{,}000
$$

optimization problems for one full episode.

### 24.5 Target splitting is heuristic

The proportional target allocation is reasonable, but it is not globally optimal. A centralized MPC could theoretically allocate target responsibilities more flexibly.

---

## 25. Reproducibility instructions

### 25.1 Run standalone script

From the controller directory:

```bash
python run_mpc.py
```

This runs the full decentralized AL-MPC episode, computes KPIs, plots results, and saves output files if `save_dir` is provided.

### 25.2 Use as a CityLearn agent

```python
from utils import make_env
from mpc_agent import MPCAgent

env = make_env(central_agent=False)
obs, _ = env.reset()

agent = MPCAgent(
    env,
    horizon=24,
    w_track=50.0,
    w_comfort=20.0,
    w_smooth=1.0,
    w_net_smooth=50.0,
    w_terminal_soc=5.0,
)

while not env.terminated:
    actions = agent.predict(obs)
    obs, _, terminated, truncated, _ = env.step(actions)
```

### 25.3 Run system identification diagnostics

```bash
python system_identification.py
```

This computes offline average LSTM Jacobians for all buildings and optionally saves them as an `.npz` archive.

---

## 26. Summary of the method in one paragraph

Adaptive LSTM-Linearized MPC (AL-MPC) is a decentralized MPC controller that uses CityLearn's pre-trained LSTM building dynamics model as a differentiable surrogate for indoor temperature evolution. At each timestep, the controller extracts local temperature and cooling-demand Jacobians through PyTorch automatic differentiation, converts them to physical units, combines them with a forecast COP model, and builds a convex linear MPC problem for each building. The objective penalizes allocated target-tracking error, thermal comfort violations, action changes, net-load oscillations, and terminal SOC deviation. The district target is decomposed across buildings proportionally to uncontrollable baseline load, enabling decentralized controllers to contribute to the global tracking task without direct communication.

---

## 27. Suggested citation name in report

Use the following name in the report and plots:

```text
Adaptive LSTM-Linearized MPC (AL-MPC)
```

First mention:

```text
The first MPC controller, referred to as Adaptive LSTM-Linearized MPC (AL-MPC), uses online Jacobian extraction from the CityLearn LSTM dynamics model to construct a local linear MPC problem for each building.
```

Short form after first mention:

```text
AL-MPC
```

---

## 28. Technical keyword list

- Decentralized MPC
- Successive linearization
- LSTM linearization
- Automatic differentiation
- Online Jacobian extraction
- Convex quadratic programming
- CVXPY
- OSQP
- Building energy management
- Battery energy storage system
- Heat pump control
- Demand response
- District load tracking
- Comfort-constrained control
- Receding-horizon optimization