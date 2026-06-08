# Coordinated Joint State-Space MPC (CJSS-MPC)

## Technical README for the former `MPC_Coordinated_Joint_SID` controller

This document describes the third MPC controller developed for the CityLearn Team Internship project.

The original internal name was:

```text
MPC_Coordinated_Joint_SID
```

For public reporting and documentation, this README uses the more professional name:

# Coordinated Joint State-Space MPC (CJSS-MPC)

This name reflects the main technical ideas in the implementation:

- **Coordinated**: a district-level coordination layer allocates the global district target across the 25 buildings.
- **Joint**: the system identification model predicts several building variables jointly, not only temperature.
- **State-Space**: the learned model predicts the next values of a compact control state.
- **MPC**: each building solves a receding-horizon optimization problem.

CJSS-MPC is the most coordinated of the MPC variants because it combines a learned multi-output system model with a district-level target allocation mechanism.

---

## 1. High-level method summary

CJSS-MPC is a hierarchical decentralized MPC architecture.

It has two levels:

1. **District coordinator**
   - predicts the baseline load trajectory of every building,
   - compares the sum of those baseline loads with the district target,
   - allocates the required correction among buildings using flexibility scores.

2. **Local building MPCs**
   - each building receives a coordinated target trajectory,
   - each building solves its own finite-horizon nonlinear MPC problem,
   - each building returns only its first battery and cooling action.

The system identification model is a learned joint residual NARX/state-space model. It predicts four next-step variables per building:

```text
indoor_dry_bulb_temperature_next
electrical_storage_soc_next
net_electricity_consumption_next
cooling_electricity_consumption_next
```

Unlike the previous HRN-MPC model, which learned only indoor temperature change and approximated the rest analytically, CJSS-MPC learns the next values of thermal, storage, and electricity-related variables jointly.

---

## 2. Recommended controller name

The recommended name is:

```text
Coordinated Joint State-Space MPC (CJSS-MPC)
```

Alternative possible names:

```text
Coordinator-Guided Joint SID-MPC
Joint Residual State-Space MPC
Coordinated Residual-NARX MPC
Hierarchical Joint SID-MPC
```

The recommended name **CJSS-MPC** is compact and report-friendly. It communicates that the method is not merely an independent per-building MPC, but includes a coordination layer and a joint learned prediction model.

---

## 3. Repository files

This controller is defined mainly by the following files.

### 3.1 `train_joint_sid.ipynb`

This notebook trains the joint system identification model.

It performs the following steps:

1. Creates the CityLearn Texas environment.
2. Collects persistently excited rollout data.
3. Builds NARX/state-history features.
4. Trains a multi-output Ridge model.
5. Trains a multi-output residual MLP ensemble.
6. Evaluates one-step validation performance.
7. Saves SID artifacts for use by the MPC controller.

The saved artifacts are expected in:

```text
saved_files/joint_sid/
```

or in the submission notebook:

```text
System Identification/joint_sid/
```

The expected artifacts are:

```text
joint_sid_bundle.joblib
joint_residual_mlp_seed_0.pt
joint_residual_mlp_seed_1.pt
joint_residual_mlp_seed_2.pt
joint_sid_training_data.parquet
```

### 3.2 `joint_sid_model.py`

This file defines the learned joint SID model and helper functions.

Important components:

```text
TARGET_NAMES
STATE_NAMES
JointResidualMLP
JointSIDModel
normalize_names
obs_to_dicts
safe_get
get_action_bounds
build_action_vector
make_time_features
build_feature_dict_from_history
```

The `JointSIDModel` class loads the trained Ridge model, scalers, feature list, and residual MLP ensemble.

### 3.3 `joint_sid_mpc_controller_coordinated.py`

This file defines the coordinated MPC controller.

Important components:

```text
JointSIDMPCConfig
JointSIDMPCController
```

The controller contains:

- history initialization,
- history update after each environment step,
- baseline rollout prediction,
- flexibility score calculation,
- coordinated target allocation,
- local building MPC objective,
- SLSQP optimization,
- CityLearn-compatible action generation.

### 3.4 `run_joint_sid_mpc_wandb.ipynb`

This notebook runs the final controller and logs results to Weights & Biases.

It:

1. Loads the Texas dataset and district target.
2. Creates the CityLearn environment.
3. Loads the joint SID model.
4. Configures CJSS-MPC.
5. Runs the controller over the simulation window.
6. Logs district load, KPIs, and temperature plots to W&B.

---

## 4. Core difference from the earlier MPC controllers

The project includes three MPC-style controllers. CJSS-MPC differs from the first two as follows.

| Aspect | AL-MPC | HRN-MPC | CJSS-MPC |
|---|---|---|---|
| Original name | `MPC_Decentralized_1` | `MPC_Advanced_SID` | `MPC_Coordinated_Joint_SID` |
| Public name | Adaptive LSTM-Linearized MPC | Hybrid Residual-NARX MPC | Coordinated Joint State-Space MPC |
| SID type | Online LSTM Jacobians | Ridge NARX + residual MLP for temperature delta | Joint Ridge + residual MLP for next state variables |
| Learned outputs | local temperature sensitivity | temperature increment | temperature, SOC, net load, cooling electricity |
| Coordinator | no | no | yes |
| Target split | proportional to baseline load | equal split | dynamic flexibility-weighted allocation |
| Optimizer | CVXPY QP | SciPy SLSQP nonlinear MPC | SciPy SLSQP nonlinear MPC |
| MPC model | linearized thermal model | nonlinear thermal model + analytical storage/load | learned multi-output state-space model |
| Architecture | decentralized | decentralized | coordinated decentralized/hierarchical |

CJSS-MPC is therefore the most integrated MPC variant because it learns multiple state variables and adds a district-level coordination mechanism.

---

## 5. Control architecture

CJSS-MPC uses a hierarchical decentralized structure.

Let there be:

```text
N_b = 25 buildings
```

At each timestep, the controller performs:

1. District coordination.
2. Per-building local MPC optimization.
3. CityLearn action construction.
4. Environment step.
5. History update.

The local action vector for building $i$ is:

$$
u_i(k) =
\begin{bmatrix}
u_{\mathrm{bat},i}(k) \\
u_{\mathrm{cool},i}(k)
\end{bmatrix}
$$

where:

- $u_{\mathrm{bat},i}(k)$ is the electrical storage action,
- $u_{\mathrm{cool},i}(k)$ is the cooling device action.

The controller is not fully centralized because it does not solve one large optimization problem over all building actions. Instead, it uses a coordinator to assign targets, then each building solves its own local problem.

---

## 6. Learned joint state-space model

The joint SID model predicts a compact next-step state vector.

The target vector is:

$$
y_i(k+1)
=
\begin{bmatrix}
T_i(k+1) \\
\mathrm{SOC}_i(k+1) \\
L_i(k+1) \\
C_i(k+1)
\end{bmatrix}
$$

where:

- $T_i(k+1)$ is next indoor dry-bulb temperature,
- $\mathrm{SOC}_i(k+1)$ is next electrical storage state of charge,
- $L_i(k+1)$ is next building net electricity consumption,
- $C_i(k+1)$ is next cooling electricity consumption.

The corresponding state names in the code are:

```text
indoor_dry_bulb_temperature
electrical_storage_soc
net_electricity_consumption
cooling_electricity_consumption
```

The target names are:

```text
indoor_dry_bulb_temperature_next
electrical_storage_soc_next
net_electricity_consumption_next
cooling_electricity_consumption_next
```

The learned model has the form:

$$
\hat{y}_i(k+1)
=
f_{\mathrm{jointSID}}(z_i(k))
$$

where $z_i(k)$ is a feature vector constructed from:

- building ID,
- time features,
- exogenous variables,
- current states,
- lagged states,
- current actions,
- lagged actions.

---

## 7. NARX/state-history feature structure

CJSS-MPC uses a NARX-like feature representation.

The model includes current and lagged values of states and actions. The lag set is:

```text
LAGS = (1, 2, 3, 6, 12)
```

For a state variable $x$, the feature vector contains:

$$
x(k), x(k-1), x(k-2), x(k-3), x(k-6), x(k-12)
$$

This gives the model access to recent thermal, storage, and load history.

The state variables with lagged histories are:

```text
indoor_dry_bulb_temperature
electrical_storage_soc
net_electricity_consumption
cooling_electricity_consumption
```

The action variables with lagged histories are:

```text
action_electrical_storage
action_cooling_device
```

This means the model can learn temporal dependencies such as:

- thermal inertia,
- delayed cooling effects,
- storage memory,
- action persistence,
- relationship between past actions and future net electricity.

---

## 8. Time feature engineering

Hour and month are cyclic variables. The SID model therefore uses sine and cosine encodings.

For hour:

$$
h_{\sin}(k) =
\sin\left(\frac{2\pi h(k)}{24}\right)
$$

$$
h_{\cos}(k) =
\cos\left(\frac{2\pi h(k)}{24}\right)
$$

For month:

$$
m_{\sin}(k) =
\sin\left(\frac{2\pi m(k)}{12}\right)
$$

$$
m_{\cos}(k) =
\cos\left(\frac{2\pi m(k)}{12}\right)
$$

This prevents artificial discontinuities between adjacent cyclic values. For example, hour 23 and hour 0 are close in time, but numerically far apart if represented as raw integers.

---

## 9. Exogenous and contextual features

The feature dictionary includes the following exogenous and contextual variables when available:

```text
outdoor_dry_bulb_temperature
direct_solar_irradiance
solar_generation
non_shiftable_load
dhw_demand
cooling_demand
heating_demand
indoor_dry_bulb_temperature_cooling_set_point
indoor_dry_bulb_temperature_heating_set_point
comfort_band
hvac_mode
power_outage
```

These variables allow the model to account for:

- weather-driven cooling load,
- solar production,
- uncontrollable plug loads,
- domestic hot water demand,
- HVAC operating regime,
- comfort schedules,
- abnormal outage conditions.

The feature builder also includes the current control actions:

```text
action_electrical_storage
action_cooling_device
```

---

## 10. Persistent-excitation data collection

The training notebook collects rollout data using piecewise-random actions.

This is important for system identification. If the controller only observes one fixed policy, the training data may not contain enough variation to learn the effect of actions. Persistent excitation forces the system to experience a wider range of inputs.

The rollout collection uses:

```text
N_ROLLOUTS = 8
LAGS = (1, 2, 3, 6, 12)
hold_prob = 0.85
SEED = 42
MAX_STEPS = None
```

### 10.1 Action sampling

For each building and action dimension:

- With probability `hold_prob = 0.85`, the previous action is repeated.
- With probability `1 - hold_prob = 0.15`, a new random action is sampled.
- Battery and cooling actions are sampled uniformly from their CityLearn action bounds.
- Inactive or unknown action variables are set to zero.

This produces piecewise-random trajectories rather than fully random white-noise actions.

Piecewise-random excitation has two advantages:

1. It excites the system sufficiently for SID.
2. It avoids unrealistic step-to-step high-frequency switching.

### 10.2 Dataset size

The Texas simulation window has approximately 720 hourly timesteps.

For 8 rollouts and 25 buildings, the raw number of building-timestep samples is approximately:

$$
8 \times 720 \times 25 = 144000
$$

The final number of training rows may be slightly smaller depending on early termination and lag/history availability.

---

## 11. Training targets

For each transition, the notebook stores:

```text
current observation
current action
next observation
```

For building $i$ at time $k$, the target vector is:

$$
y_i(k+1)
=
[
T_i(k+1),
\mathrm{SOC}_i(k+1),
L_i(k+1),
C_i(k+1)
]
$$

In code, these are:

```text
indoor_dry_bulb_temperature_next
electrical_storage_soc_next
net_electricity_consumption_next
cooling_electricity_consumption_next
```

Unlike HRN-MPC, this model does **not** only learn $\Delta T$. It directly learns multiple next-step state/output variables.

This allows the MPC to avoid approximate analytical net-load and SOC equations. Instead, those variables are predicted by the learned joint model.

---

## 12. Train/validation split

The notebook splits the data by rollout.

For the collected rollout IDs:

```text
unique_rollouts = sorted(df["rollout"].unique())
val_rollouts = last 25% of rollouts
train_rollouts = remaining 75%
```

With 8 rollouts:

```text
training rollouts: 0, 1, 2, 3, 4, 5
validation rollouts: 6, 7
```

This is better than random row splitting because it tests whether the model generalizes to unseen trajectories, not merely neighboring samples from the same trajectory.

---

## 13. Scaling

Both features and targets are standardized.

For input features:

$$
\tilde{z}
=
\frac{z - \mu_z}{\sigma_z}
$$

For target outputs:

$$
\tilde{y}
=
\frac{y - \mu_y}{\sigma_y}
$$

The scalers are saved inside the joblib bundle:

```text
x_scaler
y_scaler
```

During MPC inference:

1. feature dictionary is converted to ordered feature vector,
2. missing or invalid values are replaced with zero,
3. input vector is scaled,
4. Ridge and MLP predictions are computed in scaled space,
5. output is inverse-transformed to physical units.

---

## 14. Multi-output Ridge backbone

The first part of the model is a multi-output Ridge regression.

The model predicts the scaled target vector:

$$
\hat{\tilde{y}}_{\mathrm{ridge}}
=
B^\top \tilde{z} + b
$$

The Ridge objective is:

$$
\min_B
\sum_{k=1}^{M}
\left\|
\tilde{y}_k - B^\top \tilde{z}_k - b
\right\|_2^2
+
\alpha \|B\|_F^2
$$

The implementation uses:

```text
Ridge(alpha=1.0)
```

Since there are four outputs, the Ridge model learns a multi-output linear mapping from the feature vector to the next-state vector.

The Ridge model provides:

1. a stable linear baseline,
2. a regularized model that reduces overfitting,
3. the backbone for residual correction.

---

## 15. Joint residual MLP ensemble

The second part of the model is a neural residual ensemble.

The residual target is:

$$
r_k =
\tilde{y}_k - \hat{\tilde{y}}_{\mathrm{ridge},k}
$$

Each MLP learns:

$$
\hat{r}_k =
g_{\theta}( \tilde{z}_k )
$$

The final scaled prediction is:

$$
\hat{\tilde{y}}_k
=
\hat{\tilde{y}}_{\mathrm{ridge},k}
+
\frac{1}{S}
\sum_{s=1}^{S}
g_{\theta_s}(\tilde{z}_k)
$$

where $S = 3$ residual models.

The final physical prediction is:

$$
\hat{y}_k =
\mathrm{inverseScaler}_y(\hat{\tilde{y}}_k)
$$

### 15.1 MLP architecture

The residual MLP architecture is:

```text
Input dimension = number of retained SID features
Output dimension = 4
Hidden width = 384
Depth = 4 hidden layers
Activation = SiLU
Normalization = LayerNorm
Dropout = 0.05
```

Each hidden block is:

```text
Linear
LayerNorm
SiLU
Dropout
```

The output layer is:

```text
Linear(hidden, 4)
```

The output dimension is four because the model predicts:

```text
temperature_next
SOC_next
net_load_next
cooling_electricity_next
```

### 15.2 Training configuration

The residual MLP training uses:

```text
BATCH_SIZE = 2048
EPOCHS = 80
PATIENCE = 12
MODEL_CONFIG = {"hidden": 384, "depth": 4, "dropout": 0.05}
optimizer = AdamW
learning_rate = 2e-3
weight_decay = 1e-4
loss = MSELoss
gradient_clip_norm = 2.0
```

The model uses early stopping based on validation loss.

### 15.3 Ensemble seeds

The notebook trains three residual networks:

```text
seed = SEED + 0
seed = SEED + 1
seed = SEED + 2
```

The saved model files are:

```text
joint_residual_mlp_seed_0.pt
joint_residual_mlp_seed_1.pt
joint_residual_mlp_seed_2.pt
```

During inference, their residual outputs are averaged.

---

## 16. SID artifacts

After training, the notebook saves:

```text
joint_sid_bundle.joblib
joint_residual_mlp_seed_0.pt
joint_residual_mlp_seed_1.pt
joint_residual_mlp_seed_2.pt
joint_sid_training_data.parquet
```

The joblib bundle contains:

```text
feature_names
target_names
state_names
lags
x_scaler
y_scaler
ridge
model_config
```

The `.pt` files store the neural residual ensemble weights.

The `.parquet` file stores the rollout training data for inspection and reproducibility.

---

## 17. Joint model inference

The `JointSIDModel` class provides:

```python
predict_from_feature_dict(feature_dict)
```

This method:

1. Creates a feature row in the exact training feature order.
2. Uses zero for any missing feature.
3. Converts NaN or infinite values to zero.
4. Standardizes the feature vector.
5. Predicts the Ridge output.
6. Predicts residual corrections using all available MLPs.
7. Averages the residual predictions.
8. Adds residual prediction to Ridge prediction.
9. Inverse-transforms the result to physical units.
10. Returns a dictionary mapping target names to predicted values.

The prediction dictionary contains:

```text
indoor_dry_bulb_temperature_next
electrical_storage_soc_next
net_electricity_consumption_next
cooling_electricity_consumption_next
```

---

## 18. MPC configuration

The submitted configuration is:

```python
mpc_config = JointSIDMPCConfig(
    horizon=4,
    comfort_low=22.0,
    comfort_high=26.0,
    w_track=100.0,
    w_avg_track=500.0,
    w_ramp=100.0,
    w_comfort=20.0,
    w_smooth=50.0,
    w_action=5.0,
    w_soc=2.0,
    terminal_soc_ref=0.50,
    maxiter=80,
    verbose=False,
)
```

The default dataclass also includes coordinator settings:

```text
use_coordinator = True
coordinator_eps = 1e-6
min_flex_share = 0.02
```

### 18.1 Hyperparameter meaning

| Parameter | Value | Meaning |
|---|---:|---|
| `horizon` | 4 | prediction/control horizon in hours |
| `comfort_low` | 22.0 | lower comfort temperature bound |
| `comfort_high` | 26.0 | upper comfort temperature bound |
| `w_track` | 100.0 | pointwise local target tracking penalty |
| `w_avg_track` | 500.0 | horizon-average tracking penalty |
| `w_ramp` | 100.0 | predicted net-load ramping penalty |
| `w_comfort` | 20.0 | thermal comfort violation penalty |
| `w_smooth` | 50.0 | action smoothness penalty |
| `w_action` | 5.0 | action magnitude penalty |
| `w_soc` | 2.0 | terminal SOC penalty |
| `terminal_soc_ref` | 0.50 | desired final SOC over horizon |
| `maxiter` | 80 | maximum SLSQP iterations |
| `use_coordinator` | True | enables district-level dynamic target allocation |
| `min_flex_share` | 0.02 | avoids zero flexibility weights |
| `coordinator_eps` | 1e-6 | numerical stabilizer in weight normalization |

---

## 19. History initialization

Before the first MPC step, the controller initializes histories from the current observation.

For each building and state variable, the history is initialized by repeating the current value:

```text
max_lag + 1 times
```

Since:

```text
max_lag = 12
```

each history starts with 13 repeated values.

This ensures that lag features such as $x(k-12)$ are available from the first control step.

Action histories are initialized to zero:

```text
action_electrical_storage = 0
action_cooling_device = 0
```

---

## 20. History update after each environment step

After applying actions and stepping the CityLearn environment, the controller calls:

```python
update_histories_after_step(obs, actions)
```

This appends the newest observed values for:

```text
indoor_dry_bulb_temperature
electrical_storage_soc
net_electricity_consumption
cooling_electricity_consumption
```

and the newest applied actions:

```text
action_electrical_storage
action_cooling_device
```

The history is trimmed to:

```text
max_lag + 5
```

This keeps enough context for lag features while avoiding unbounded memory growth.

---

## 21. Recursive building rollout model

The method `_rollout_building` simulates a building over the MPC horizon using the learned joint SID model.

Given:

- building index $i$,
- current observation dictionary,
- current state/action history,
- candidate action sequence $U_i$,

it recursively predicts:

$$
\hat{y}_i(k+1), \hat{y}_i(k+2), \ldots, \hat{y}_i(k+H)
$$

At each horizon step:

1. Construct a feature dictionary from current simulated state/history and candidate action.
2. Call the joint SID model.
3. Extract predicted temperature, SOC, net load, and cooling electricity.
4. Clip temperature and SOC to numerical ranges.
5. Append predicted values into history.
6. Use updated history for the next horizon step.

The prediction recursion is important. The model does not merely predict one-step ahead using true future observations. It feeds its own predicted states back into the features.

---

## 22. Numerical clipping inside rollout

The predicted indoor temperature is clipped to:

```text
10 °C <= T <= 45 °C
```

The predicted SOC is clipped to:

```text
0 <= SOC <= 1
```

Cooling electricity is clipped from below:

```text
cooling electricity >= 0
```

These are numerical safety measures. They prevent extreme predictions from destabilizing the optimizer. They do not directly impose hard constraints on the real CityLearn environment.

---

## 23. District coordinator

The district coordinator is the main architectural addition in CJSS-MPC.

The coordinator does not directly optimize all building actions. Instead, it creates coordinated local target trajectories.

The process is:

1. Predict baseline load trajectory for each building.
2. Sum building baselines to obtain district baseline.
3. Compare district baseline with district target.
4. Compute required correction.
5. Distribute the correction among buildings using flexibility weights.
6. Send each building its coordinated target trajectory.

---

## 24. Baseline rollout

For each building, the coordinator first predicts what would happen if the previous action were repeated over the horizon.

For building $i$:

$$
U_i^{\mathrm{base}}
=
[u_i(k-1), u_i(k-1), \ldots, u_i(k-1)]
$$

The learned model predicts the corresponding net load trajectory:

$$
\hat{L}_{i}^{\mathrm{base}}(k+h)
$$

The district baseline is:

$$
\hat{L}_{\mathrm{district}}^{\mathrm{base}}(k+h)
=
\sum_{i=1}^{N_b}
\hat{L}_{i}^{\mathrm{base}}(k+h)
$$

The district error is:

$$
e_{\mathrm{district}}(k+h)
=
r_{\mathrm{district}}(k+h)
-
\hat{L}_{\mathrm{district}}^{\mathrm{base}}(k+h)
$$

where $r_{\mathrm{district}}$ is the district target.

If the error is positive, the district needs more load than the baseline.  
If the error is negative, the district needs less load than the baseline.

---

## 25. Flexibility scores

The coordinator estimates each building's ability to increase or decrease load.

For building $i$:

### 25.1 Upward flexibility

Upward flexibility means the building can increase net load.

The implementation uses:

$$
F_i^{\uparrow}
=
F_{\mathrm{battery},i}^{\uparrow}
+
F_{\mathrm{cooling},i}^{\uparrow}
+
F_{\min}
$$

Battery upward flexibility:

$$
F_{\mathrm{battery},i}^{\uparrow}
=
\max(0, u_{\mathrm{bat},i}^{\max})
\max(0, 1 - \mathrm{SOC}_i)
$$

Cooling upward flexibility:

$$
F_{\mathrm{cooling},i}^{\uparrow}
=
(u_{\mathrm{cool}}^{\max} - u_{\mathrm{cool}}^{\min})
\frac{\max(0, T_i - T_{\min})}{T_{\max} - T_{\min}}
$$

where:

- $T_{\min} = 22^\circ C$,
- $T_{\max} = 26^\circ C$,
- $F_{\min} = 0.02$.

Intuition:

- A battery can increase load by charging if it has room below full SOC.
- Cooling can increase load if increasing cooling is thermally safe.

### 25.2 Downward flexibility

Downward flexibility means the building can reduce net load.

The implementation uses:

$$
F_i^{\downarrow}
=
F_{\mathrm{battery},i}^{\downarrow}
+
F_{\mathrm{cooling},i}^{\downarrow}
+
F_{\min}
$$

Battery downward flexibility:

$$
F_{\mathrm{battery},i}^{\downarrow}
=
\max(0, -u_{\mathrm{bat},i}^{\min})
\max(0, \mathrm{SOC}_i)
$$

Cooling downward flexibility:

$$
F_{\mathrm{cooling},i}^{\downarrow}
=
(u_{\mathrm{cool}}^{\max} - u_{\mathrm{cool}}^{\min})
\frac{\max(0, T_{\max} - T_i)}{T_{\max} - T_{\min}}
$$

Intuition:

- A battery can reduce load by discharging if it has available charge.
- Cooling can be reduced if the building is not too warm.

---

## 26. Coordinated target allocation

At each horizon step, the district error is allocated using flexibility weights.

If the district needs more load:

$$
w_i(k+h) =
\frac{F_i^{\uparrow}}
{\sum_{j=1}^{N_b} F_j^{\uparrow} + \epsilon}
$$

If the district needs less load:

$$
w_i(k+h) =
\frac{F_i^{\downarrow}}
{\sum_{j=1}^{N_b} F_j^{\downarrow} + \epsilon}
$$

where:

```text
epsilon = coordinator_eps = 1e-6
```

The coordinated local target is:

$$
r_i^{\mathrm{coord}}(k+h)
=
\hat{L}_i^{\mathrm{base}}(k+h)
+
w_i(k+h)
e_{\mathrm{district}}(k+h)
$$

This construction guarantees that, approximately:

$$
\sum_{i=1}^{N_b}
r_i^{\mathrm{coord}}(k+h)
\approx
r_{\mathrm{district}}(k+h)
$$

because the correction weights sum to approximately one.

If the coordinator is disabled, the controller falls back to equal splitting:

$$
r_i(k+h) =
\frac{r_{\mathrm{district}}(k+h)}
{N_b}
$$

---

## 27. Local MPC decision variables

For each building $i$, the local MPC chooses a sequence of actions over horizon $H$:

$$
U_i =
[
u_i(k),
u_i(k+1),
\ldots,
u_i(k+H-1)
]
$$

Each action has two components:

$$
u_i(k+h)
=
[
u_{\mathrm{bat},i}(k+h),
u_{\mathrm{cool},i}(k+h)
]
$$

The flattened optimization vector has dimension:

$$
2H
$$

For the submitted configuration:

```text
H = 4
```

so each building solves an 8-variable nonlinear optimization problem.

Because there are 25 buildings, each simulation timestep requires 25 local SLSQP optimizations.

---

## 28. Local MPC objective

For building $i$, the local objective contains seven terms:

1. pointwise tracking,
2. average tracking,
3. ramping,
4. comfort,
5. action smoothness,
6. action magnitude,
7. terminal SOC.

The implemented objective is:

$$
J_i =
w_{\mathrm{track}} J_{\mathrm{track}}
+
w_{\mathrm{avg}} J_{\mathrm{avg}}
+
w_{\mathrm{ramp}} J_{\mathrm{ramp}}
+
w_{\mathrm{comfort}} J_{\mathrm{comfort}}
+
w_{\mathrm{smooth}} J_{\mathrm{smooth}}
+
w_{\mathrm{action}} J_{\mathrm{action}}
+
w_{\mathrm{soc}} J_{\mathrm{soc}}
$$

where all terms are computed over the horizon.

---

## 29. Pointwise tracking term

The pointwise tracking term is:

$$
J_{\mathrm{track}}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\left(
\hat{L}_i(k+h) -
r_i^{\mathrm{coord}}(k+h)
\right)^2
$$

This penalizes hour-by-hour deviation from the coordinated local target.

Submitted weight:

```text
w_track = 100.0
```

---

## 30. Average tracking term

The average tracking term is:

$$
J_{\mathrm{avg}}
=
\left(
\frac{1}{H}
\sum_{h=0}^{H-1}
\hat{L}_i(k+h)
-
\frac{1}{H}
\sum_{h=0}^{H-1}
r_i^{\mathrm{coord}}(k+h)
\right)^2
$$

This term encourages the average predicted load over the horizon to match the average assigned target.

Submitted weight:

```text
w_avg_track = 500.0
```

This is the largest weight in the controller. It emphasizes matching the average target level over the short horizon.

---

## 31. Ramping term

The ramping term is:

$$
J_{\mathrm{ramp}}
=
\frac{1}{H-1}
\sum_{h=1}^{H-1}
\left(
\hat{L}_i(k+h) -
\hat{L}_i(k+h-1)
\right)^2
$$

This penalizes sharp predicted changes in building net electricity consumption.

Submitted weight:

```text
w_ramp = 100.0
```

This term is especially important because the learned net-load prediction can otherwise lead to noisy action choices.

---

## 32. Comfort term

Thermal comfort is enforced softly.

The comfort range is:

```text
22 °C <= T <= 26 °C
```

Lower violation:

$$
v_i^{\mathrm{low}}(k+h)
=
\max(0, 22 - \hat{T}_i(k+h))
$$

Upper violation:

$$
v_i^{\mathrm{high}}(k+h)
=
\max(0, \hat{T}_i(k+h) - 26)
$$

Comfort cost:

$$
J_{\mathrm{comfort}}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\left[
(v_i^{\mathrm{low}}(k+h))^2
+
(v_i^{\mathrm{high}}(k+h))^2
\right]
$$

Submitted weight:

```text
w_comfort = 20.0
```

Comfort is not a hard constraint in this implementation. The optimizer can choose actions that violate comfort if other cost terms dominate.

---

## 33. Action smoothness term

The action smoothness term penalizes changes in battery and cooling actions:

$$
J_{\mathrm{smooth}}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\left\|
u_i(k+h) - u_i(k+h-1)
\right\|_2^2
$$

For $h=0$, the previous real action is used:

$$
u_i(k-1)
$$

Submitted weight:

```text
w_smooth = 50.0
```

This strongly discourages rapid switching.

---

## 34. Action magnitude term

The action magnitude penalty is:

$$
J_{\mathrm{action}}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\left\|
u_i(k+h)
\right\|_2^2
$$

Submitted weight:

```text
w_action = 5.0
```

This discourages unnecessarily large battery and cooling actions.

---

## 35. Terminal SOC term

The terminal SOC penalty is:

$$
J_{\mathrm{soc}}
=
\left(
\hat{\mathrm{SOC}}_i(k+H) - 0.5
\right)^2
$$

Submitted weight:

```text
w_soc = 2.0
terminal_soc_ref = 0.50
```

This prevents the optimizer from always draining or filling the battery inside the short horizon.

---

## 36. Bounds and constraints

The optimizer uses action bounds from the CityLearn environment.

For each building:

$$
u_{\mathrm{bat}}^{\min}
\leq
u_{\mathrm{bat}}(k+h)
\leq
u_{\mathrm{bat}}^{\max}
$$

$$
u_{\mathrm{cool}}^{\min}
\leq
u_{\mathrm{cool}}(k+h)
\leq
u_{\mathrm{cool}}^{\max}
$$

These bounds are extracted by:

```python
get_action_bounds(env, action_names)
```

No explicit hard constraints are added for comfort, SOC, or net load. Instead:

- SOC is clipped in the predicted rollout,
- temperature is clipped for numerical stability,
- comfort violations are penalized in the objective,
- action feasibility is handled through bounds.

---

## 37. Optimizer

The controller uses:

```text
scipy.optimize.minimize
method = SLSQP
maxiter = 80
ftol = 1e-4
```

The optimization is nonlinear because the objective depends on the recursive learned neural model.

The problem is not convex because:

1. the residual MLP is nonlinear,
2. predictions are recursively fed back into future features,
3. coordinator targets depend on learned baseline rollouts,
4. the objective includes nonlinear comfort and ramping terms.

If the optimizer fails, the controller uses the repeated previous action sequence as fallback.

---

## 38. Initial guess and warm start

For each building, the initial optimization vector is constructed by repeating the previous action over the horizon:

$$
x_0 =
[
u_i(k-1),
u_i(k-1),
\ldots,
u_i(k-1)
]
$$

This acts as a basic warm start. It also encourages action continuity.

If the previous action is outside the current action bounds, it is clipped.

---

## 39. Receding horizon operation

CJSS-MPC applies only the first action from the optimized sequence.

At timestep $k$:

1. Optimize sequence $U_i(k:k+H-1)$.
2. Extract first action $u_i(k)$.
3. Apply $u_i(k)$ to CityLearn.
4. Observe next state.
5. Update history.
6. Repeat at $k+1$.

This is standard MPC behavior.

---

## 40. Full online algorithm

```text
1. Load district target for June.
2. Create decentralized CityLearn environment.
3. Load JointSIDModel artifacts.
4. Initialize JointSIDMPCController.
5. At the first control step:
      a. initialize all state/action histories from observations.
6. At every timestep:
      a. convert observations to dictionaries.
      b. predict baseline horizon loads by repeating previous actions.
      c. compute district baseline load.
      d. compute district error relative to target.
      e. compute each building's upward/downward flexibility.
      f. allocate district error across buildings.
      g. for each building:
             i. solve local nonlinear MPC with coordinated target.
            ii. return first battery/cooling action.
      h. apply all actions to CityLearn.
      i. update histories using the new observations and applied actions.
      j. log district load to W&B.
7. After the episode:
      a. compute KPIs.
      b. create temperature comfort plot.
      c. log summary metrics to W&B.
```

---

## 41. W&B submission workflow

The run notebook uses:

```text
WANDB_ENTITY = CityLearn-TeamB
WANDB_PROJECT = CityLearn
run name = MPC_Coordinated_Joint_SID
```

During the simulation, the notebook logs:

```text
hour
district_load
```

After the simulation, it logs:

```text
NMBE [%]
CV-RMSE [%]
Temp Comfort violation [%]
temperature_plot
```

The temperature plot includes:

- all building temperatures,
- mean building temperature,
- comfort band from 22 °C to 26 °C.

---

## 42. KPI evaluation

The controller is evaluated against the Texas district target for the June simulation window:

```text
simulation_start = 3624
simulation_end = 4343
T = 720 hours
```

The same project KPI function is used:

```python
compute_kpis(district_target[:len(district_load_mpc)], district_load_mpc, mpc_building_temps)
```

The KPIs are:

```text
NMBE [%]
CV-RMSE [%]
Temp Comfort violation [%]
```

---

## 43. Strengths of CJSS-MPC

### 43.1 Learns multiple outputs jointly

The model predicts temperature, SOC, net load, and cooling electricity together. This can capture correlations between variables that separate models may miss.

### 43.2 Avoids hand-coded electricity balance inside MPC

Unlike HRN-MPC, which approximates net load algebraically, CJSS-MPC directly predicts net electricity consumption.

### 43.3 Includes district-level coordination

The coordinator aligns local building targets with the district-level target. This is more sophisticated than equal target splitting.

### 43.4 Uses flexibility-aware allocation

Buildings with more available upward or downward flexibility receive larger target corrections.

### 43.5 Uses average tracking and ramp penalties

The local MPC objective does not only track pointwise values. It also penalizes horizon-average mismatch and predicted load ramping.

### 43.6 Fully decentralized optimization after coordination

The coordinator creates targets, but each building solves its own smaller local optimization problem.

---

## 44. Limitations of CJSS-MPC

### 44.1 Learned model errors affect all predicted variables

Because SOC, net load, and cooling electricity are learned, errors in the joint model can directly affect the optimization.

### 44.2 Nonconvex optimization

The residual MLP makes the objective nonlinear and nonconvex. SLSQP may find only a local solution.

### 44.3 Computational cost

Each timestep requires:

```text
25 local optimizations
```

Each local optimization repeatedly performs recursive learned-model rollouts.

### 44.4 Short prediction horizon

The submitted horizon is only:

```text
H = 4 hours
```

This improves runtime but limits long-term battery scheduling.

### 44.5 Coordinator uses heuristic flexibility scores

The coordinator's flexibility score is not learned or optimized globally. It is a physically motivated heuristic.

### 44.6 Soft comfort handling

Comfort is penalized but not enforced as a hard constraint.

### 44.7 Baseline depends on repeated previous actions

The coordinator estimates baseline using repeated previous actions, which may not always represent what the building would naturally do under future disturbances.

---

## 45. Recommended future improvements

1. Increase the SID rollout count beyond 8 for stronger model generalization.
2. Add recursive multi-step training loss to reduce rollout drift.
3. Use uncertainty estimates from the residual ensemble.
4. Warm-start SLSQP using the shifted previous optimal trajectory.
5. Replace SLSQP with a faster differentiable optimizer.
6. Add hard comfort constraints or stronger adaptive comfort penalties.
7. Improve coordinator flexibility scoring using learned sensitivity analysis.
8. Use proportional target allocation as a backup to coordinator allocation.
9. Train climate-specific or building-cluster-specific joint SID models.
10. Compare coordinator-enabled vs coordinator-disabled operation explicitly.

---

## 46. Minimal usage example

```python
from citylearn.citylearn import CityLearnEnv
from joint_sid_model import JointSIDModel
from joint_sid_mpc_controller_coordinated import JointSIDMPCController, JointSIDMPCConfig

env = CityLearnEnv(
    schema=str(SCHEMA_PATH),
    root_directory=str(DATASET_DIR),
    central_agent=False
)

obs, _ = env.reset()

sid_model = JointSIDModel(SID_DIR)

config = JointSIDMPCConfig(
    horizon=4,
    comfort_low=22.0,
    comfort_high=26.0,
    w_track=100.0,
    w_avg_track=500.0,
    w_ramp=100.0,
    w_comfort=20.0,
    w_smooth=50.0,
    w_action=5.0,
    w_soc=2.0,
    terminal_soc_ref=0.50,
    maxiter=80,
    verbose=False,
)

controller = JointSIDMPCController(
    env=env,
    sid_model=sid_model,
    district_target=district_target,
    config=config,
)

while not env.terminated:
    actions = controller.predict(obs)
    obs, _, terminated, truncated, _ = env.step(actions)
    controller.update_histories_after_step(obs, actions)
    if terminated or truncated:
        break
```

---

## 47. Final conceptual summary

Coordinated Joint State-Space MPC is a hierarchical MPC architecture for district load tracking in CityLearn.

It uses a learned joint residual state-space model to predict:

```text
temperature,
battery SOC,
net electricity consumption,
cooling electricity consumption.
```

A coordinator first estimates the district-level tracking error and distributes it across buildings using flexibility scores. Then, each building independently solves a nonlinear local MPC problem to track its assigned coordinated target while balancing comfort, ramping, action smoothness, action magnitude, and terminal SOC.

The central idea is:

```text
learn the relevant building state transitions jointly,
coordinate local targets at the district level,
and solve smaller local MPC problems instead of one large centralized problem.
```

This makes CJSS-MPC a bridge between decentralized learned MPC and fully centralized district optimization.
