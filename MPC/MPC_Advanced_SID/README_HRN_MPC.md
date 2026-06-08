# Hybrid Residual-NARX Model Predictive Control (HRN-MPC)

## Technical README for the former `MPC_Advanced_SID` controller

This document describes the second MPC controller developed for the CityLearn Team Internship project.  
The original working name of the controller was `MPC_Advanced_SID`. For public documentation and reporting, this README uses the more descriptive name:

# Hybrid Residual-NARX MPC (HRN-MPC)

The name reflects the main technical structure of the method:

- **Hybrid**: combines data-driven thermal modeling with analytical approximations for storage and electricity balance.
- **Residual**: uses a neural residual correction on top of a linear baseline model.
- **NARX**: uses nonlinear autoregressive features with exogenous inputs, including lagged indoor temperatures, weather, actions, and load variables.
- **MPC**: solves a receding-horizon optimization problem at each control step.

In short, HRN-MPC is a decentralized model predictive controller where each building solves its own local optimization problem using a learned thermal system identification model.

---

## 1. High-level summary

HRN-MPC is a decentralized control architecture for the Texas CityLearn district. Each of the 25 residential buildings has its own MPC controller. The controller uses:

1. A learned thermal system identification model for indoor temperature prediction.
2. Approximate analytical battery state-of-charge dynamics.
3. Approximate algebraic electricity balance.
4. A local per-building share of the district-level target load.
5. A nonlinear numerical optimizer to choose battery and cooling actions.

The thermal dynamics are learned offline from simulated rollout data. The learned model predicts the indoor temperature change:

$$
\Delta T_k = T_{k+1} - T_k
$$

Instead of predicting the absolute next temperature directly, the model predicts the temperature increment. The next temperature is then reconstructed as:

$$
T_{k+1} = T_k + \Delta T_k
$$

This is useful because temperature increments are usually smoother and easier to model than absolute temperature values.

---

## 2. Controller naming

The original internal controller name was:

```text
MPC_Advanced_SID
```

A more appropriate name for public use is:

```text
Hybrid Residual-NARX MPC (HRN-MPC)
```

Alternative names that would also be reasonable:

```text
Residual-NARX MPC
Learned Thermal SID-MPC
Hybrid SID-MPC
Residual Ensemble MPC
```

The recommended name is **Hybrid Residual-NARX MPC (HRN-MPC)** because it describes both the system identification method and the control formulation.

---

## 3. Repository files

This controller is mainly defined by the following files.

### 3.1 `CityLearn_Advanced_SID_Notebook.ipynb`

This notebook performs the advanced system identification step. It:

1. Creates the CityLearn Texas environment.
2. Collects persistently exciting rollout data.
3. Constructs NARX-style features.
4. Trains a linear Ridge baseline model.
5. Trains a residual MLP ensemble on the remaining prediction error.
6. Evaluates both one-step and recursive rollout prediction performance.
7. Saves the trained SID artifacts.

Expected saved artifacts:

```text
System Identification/advanced_sid/sid_preprocessing_and_ridge.joblib
System Identification/advanced_sid/residual_mlp_seed_0.pt
System Identification/advanced_sid/residual_mlp_seed_1.pt
System Identification/advanced_sid/residual_mlp_seed_2.pt
```

### 3.2 `sid_mpc_controller.py`

This file contains the deployed controller implementation. It defines:

```text
ResidualMLP
LearnedThermalSID
SIDMPCConfig
SIDMPCController
```

The controller loads the trained SID artifacts and solves a local nonlinear MPC problem for each building at each timestep.

### 3.3 `mpc_advanced_sid_submission.ipynb`

This notebook is the evaluation and submission workflow. It:

1. Loads the Texas CityLearn environment.
2. Loads the trained SID model.
3. Configures the MPC hyperparameters.
4. Runs the controller over the June simulation window.
5. Computes KPIs.
6. Logs results to Weights & Biases.

The submitted run used the name:

```text
MPC_Advanced_SID
```

but this README refers to the method as **HRN-MPC**.

---

## 4. Main difference from the previous AL-MPC controller

The earlier **Adaptive LSTM-Linearized MPC (AL-MPC)** used CityLearn's internal LSTM building model and differentiated through it online to obtain local Jacobians. It then solved a convex quadratic program.

HRN-MPC is different:

| Aspect | AL-MPC | HRN-MPC |
|---|---|---|
| Thermal model | CityLearn internal LSTM linearized online | Offline learned Ridge NARX + residual MLP ensemble |
| System identification | Automatic differentiation through existing LSTM | Data-driven rollout collection and supervised learning |
| MPC model type | Linearized model | Nonlinear learned prediction model |
| Optimizer | CVXPY, OSQP/SCS | SciPy `minimize`, mainly SLSQP |
| Optimization type | Convex QP | Nonlinear constrained/bounded optimization |
| Target split | Proportional to uncontrollable load | Equal split across buildings |
| Thermal prediction | Linear one-step model with disturbance estimate | Recursive nonlinear temperature-increment model |
| Main strength | Convex and interpretable | More expressive learned thermal model |
| Main limitation | Dependent on LSTM Jacobian quality | Slower and nonconvex due to nonlinear prediction model |

---

## 5. Control architecture

HRN-MPC is decentralized.

Let there be $N_b = 25$ buildings. Each building $i$ has its own MPC controller. The controller for building $i$ only produces actions for that building:

$$
u_i(k) = [u_{\text{bat},i}(k), u_{\text{cool},i}(k)]
$$

where:

- $u_{\text{bat},i}(k)$ is the electrical storage action.
- $u_{\text{cool},i}(k)$ is the cooling device action.

The controller does not solve one large district-level optimization problem. Instead, the district target is divided into local per-building targets. Each building tries to track its own local target share.

The submitted implementation uses an equal target split:

$$
r_i(k) = \frac{r_{\text{district}}(k)}{N_b}
$$

where:

- $r_{\text{district}}(k)$ is the district target load at timestep $k$.
- $r_i(k)$ is the local target assigned to building $i$.
- $N_b = 25$.

This is simpler than proportional target allocation and keeps the controller easy to inspect.

---

## 6. System identification overview

The system identification component learns a model of indoor temperature dynamics.

The target is:

$$
\Delta T_k = T_{k+1} - T_k
$$

The learned prediction model has the form:

$$
\widehat{\Delta T}_k =
f_{\text{SID}}(z_k)
$$

where $z_k$ is a feature vector containing:

- building identity,
- time features,
- weather features,
- current indoor temperature,
- lagged indoor temperatures,
- setpoints and comfort information,
- storage state of charge,
- loads and solar generation,
- current control actions.

The predicted next temperature is:

$$
\widehat{T}_{k+1}
=
T_k + \widehat{\Delta T}_k
$$

The model is called a NARX-style model because it uses autoregressive output lags and exogenous inputs:

$$
\widehat{\Delta T}_k =
f(T_k, T_{k-1}, T_{k-2}, T_{k-3}, T_{k-6}, T_{k-12}, u_k, w_k)
$$

where:

- $T_k$ is current indoor temperature,
- $T_{k-\ell}$ are lagged indoor temperatures,
- $u_k$ are control actions,
- $w_k$ are exogenous variables such as weather, solar, loads, and time features.

---

## 7. Persistent excitation data collection

The SID notebook does not train only from a zero-action trajectory or one fixed RBC trajectory. Instead, it collects data using piecewise-random actions.

This is important because a useful control model must observe how the system responds to different inputs. If all actions are constant or zero, the model cannot learn the control influence properly.

The data collection policy uses:

```text
N_ROLLOUTS = 25
hold_min = 2
hold_max = 8
zero_prob = 0.10
cooling_bias = 0.55
```

### 7.1 Piecewise-random action logic

At the start of each action segment, a random action is sampled for every building. The same action is then held for a random number of hours between 2 and 8.

This produces smoother excitation than fully random actions at every hour. Fully random actions would create unrealistic high-frequency switching. Piecewise-constant excitation is more representative of practical control while still producing enough variation for SID.

At each action sampling event:

- With probability 0.10, an action is set to zero.
- Cooling actions are sampled from their allowed bounds.
- Battery actions are sampled from their allowed bounds.
- Cooling is slightly biased toward positive cooling values if the action range is symmetric.

The data collection process records, for every rollout, timestep, and building:

- current observations,
- applied actions,
- next indoor temperature,
- next net electricity consumption,
- next cooling electricity consumption,
- next battery state of charge.

The collected dataset therefore has the structure:

```text
rollout, timestep, building_id, observations, actions, next observations
```

For the submitted SID configuration:

```text
25 rollouts × approximately 720 steps × 25 buildings
```

which gives approximately:

```text
25 × 720 × 25 = 450,000 building-timestep samples
```

before dropping rows with missing lag features.

---

## 8. Recorded variables during SID rollout

The SID rollout records a set of control-oriented variables.

### 8.1 Base observation variables

The recorded observations include:

```text
month
hour
outdoor_dry_bulb_temperature
direct_solar_irradiance
outdoor_dry_bulb_temperature_predicted_1
outdoor_dry_bulb_temperature_predicted_2
outdoor_dry_bulb_temperature_predicted_3
direct_solar_irradiance_predicted_1
direct_solar_irradiance_predicted_2
direct_solar_irradiance_predicted_3
indoor_dry_bulb_temperature
non_shiftable_load
dhw_demand
solar_generation
cooling setpoint
heating setpoint
comfort_band
hvac_mode
power_outage
electrical_storage_soc
net_electricity_consumption
cooling_electricity_consumption
heating_electricity_consumption
dhw_electricity_consumption
electrical_storage_electricity_consumption
```

### 8.2 Action variables

The recorded action variables include:

```text
action_cooling_device
action_electrical_storage
```

Depending on the active action names in the CityLearn environment, additional inactive action variables may exist, but the deployed MPC mainly uses battery and cooling actions.

### 8.3 Prediction targets

The SID model mainly targets:

```text
target_delta_T = next_indoor_dry_bulb_temperature - indoor_dry_bulb_temperature
```

It also stores:

```text
target_next_T
target_next_net_load
```

The next net load surrogate is only diagnostic. The actual MPC uses an approximate algebraic electricity balance instead of relying on a learned net-load model.

---

## 9. Feature engineering

### 9.1 Time features

Hour and month are cyclic variables. The notebook therefore transforms them using sine and cosine functions.

For hour:

$$
h_{\sin}(k) = \sin\left(\frac{2\pi h(k)}{24}\right)
$$

$$
h_{\cos}(k) = \cos\left(\frac{2\pi h(k)}{24}\right)
$$

For month:

$$
m_{\sin}(k) = \sin\left(\frac{2\pi (m(k)-1)}{12}\right)
$$

$$
m_{\cos}(k) = \cos\left(\frac{2\pi (m(k)-1)}{12}\right)
$$

This avoids treating hour 24 and hour 1 as far apart.

### 9.2 Temperature lag features

The model uses lagged indoor temperatures:

```text
lags = (1, 2, 3, 6, 12)
```

For each building trajectory, the following features are added:

```text
indoor_dry_bulb_temperature_lag1
indoor_dry_bulb_temperature_lag2
indoor_dry_bulb_temperature_lag3
indoor_dry_bulb_temperature_lag6
indoor_dry_bulb_temperature_lag12
```

The feature vector therefore includes thermal memory over multiple time scales:

- 1 hour,
- 2 hours,
- 3 hours,
- 6 hours,
- 12 hours.

This lets the model approximate building thermal inertia.

### 9.3 Final candidate feature set

The candidate features include:

```text
building_id
hour_sin
hour_cos
month_sin
month_cos
outdoor_dry_bulb_temperature
direct_solar_irradiance
outdoor_dry_bulb_temperature_predicted_1
outdoor_dry_bulb_temperature_predicted_2
outdoor_dry_bulb_temperature_predicted_3
direct_solar_irradiance_predicted_1
direct_solar_irradiance_predicted_2
direct_solar_irradiance_predicted_3
indoor_dry_bulb_temperature
indoor_dry_bulb_temperature_cooling_set_point
indoor_dry_bulb_temperature_heating_set_point
comfort_band
hvac_mode
power_outage
electrical_storage_soc
non_shiftable_load
dhw_demand
solar_generation
action_cooling_device
action_electrical_storage
temperature lag features
```

Only features that exist in the actual collected dataframe are retained.

---

## 10. Train, validation, and test split

The SID dataset is split by rollout, not by random rows.

This is important because random row splitting would leak temporal information. If consecutive samples from the same trajectory appear in both training and test data, the evaluation becomes overly optimistic.

The notebook uses:

```text
train_rollouts = all rollouts except the last two
validation_rollout = second-to-last rollout
test_rollout = last rollout
```

For 25 rollouts:

```text
training rollouts: 0 to 22
validation rollout: 23
test rollout: 24
```

This gives a trajectory-level generalization test.

---

## 11. Linear Ridge NARX baseline

The first SID model is a linear Ridge regression model.

The feature matrix is standardized:

$$
\tilde{z}_k =
\frac{z_k - \mu_z}{\sigma_z}
$$

The target temperature increment is also standardized:

$$
\tilde{y}_k =
\frac{\Delta T_k - \mu_y}{\sigma_y}
$$

The Ridge model predicts:

$$
\widehat{\tilde{\Delta T}}_{\text{ridge},k}
=
\beta_0 + \beta^\top \tilde{z}_k
$$

The Ridge objective is:

$$
\min_{\beta}
\sum_{k=1}^{M}
\left(
\tilde{y}_k - \beta_0 - \beta^\top \tilde{z}_k
\right)^2
+
\alpha \lVert \beta \rVert_2^2
$$

The implementation uses:

```text
Ridge(alpha=1.0)
```

This Ridge model serves two purposes:

1. It is a classical control-oriented linear NARX baseline.
2. It provides the linear component of the final hybrid residual model.

---

## 12. Residual MLP ensemble

The advanced SID model improves the Ridge baseline using neural residual learning.

The final model is:

$$
\widehat{\Delta T}_k
=
\widehat{\Delta T}_{\text{ridge},k}
+
\widehat{\Delta T}_{\text{residual},k}
$$

In scaled target space, the neural network learns:

$$
r_k =
\tilde{\Delta T}_k
-
\widehat{\tilde{\Delta T}}_{\text{ridge},k}
$$

The MLP is trained to predict $r_k$ from the same scaled feature vector.

### 12.1 Residual architecture

Each residual model has the following architecture:

```text
Input dimension = number of retained features
Hidden width = 256
Depth = 4 hidden layers
Activation = SiLU
Normalization = LayerNorm
Dropout = 0.05
Output dimension = 1
```

Layer structure:

```text
Linear(input_dim, 256)
LayerNorm(256)
SiLU
Dropout(0.05)

Linear(256, 256)
LayerNorm(256)
SiLU
Dropout(0.05)

Linear(256, 256)
LayerNorm(256)
SiLU
Dropout(0.05)

Linear(256, 256)
LayerNorm(256)
SiLU
Dropout(0.05)

Linear(256, 1)
```

### 12.2 Training configuration

The residual MLP training uses:

```text
optimizer = AdamW
learning_rate = 2e-3
weight_decay = 1e-4
batch_size = 2048
epochs = 250
early_stopping_patience = 30
gradient_clip_norm = 5.0
scheduler = ReduceLROnPlateau(factor=0.5, patience=8)
loss = MSE
```

### 12.3 Ensemble

The submitted SID trains three residual models with different random seeds:

```text
ENSEMBLE_SEEDS = [0, 1, 2]
```

The final residual prediction is the average of the ensemble:

$$
\widehat{r}_k =
\frac{1}{3}
\sum_{s=1}^{3}
f_{\theta_s}(\tilde{z}_k)
$$

The final scaled prediction is:

$$
\widehat{\tilde{\Delta T}}_k =
\widehat{\tilde{\Delta T}}_{\text{ridge},k}
+
\widehat{r}_k
$$

The prediction is then transformed back to physical units using the inverse target scaler:

$$
\widehat{\Delta T}_k =
\sigma_y \widehat{\tilde{\Delta T}}_k + \mu_y
$$

---

## 13. SID artifact saving

After training, the notebook saves:

```text
sid_preprocessing_and_ridge.joblib
residual_mlp_seed_0.pt
residual_mlp_seed_1.pt
residual_mlp_seed_2.pt
```

The `.joblib` bundle contains:

```text
features
lags
x_scaler
y_scaler
ridge
one_step_table
rollout_table
```

The `.pt` files contain PyTorch weights for the residual MLP ensemble.

The deployed MPC does not retrain the SID model. It only loads these saved artifacts.

---

## 14. SID model loading in the MPC

The `LearnedThermalSID` class loads the saved artifacts.

It performs the following steps:

1. Load preprocessing and Ridge model from `sid_preprocessing_and_ridge.joblib`.
2. Read the feature list and lag configuration.
3. Load the input scaler.
4. Load the target scaler.
5. Load the Ridge model.
6. Search for residual MLP files matching `residual_mlp_seed_*.pt`.
7. Instantiate one MLP per saved seed.
8. Load weights into each MLP.
9. Put all models in evaluation mode.

If no residual MLP files are found, the controller falls back to Ridge-only prediction.

The prediction function is:

```python
predict_delta_T(feature_dict)
```

It:

1. Builds a feature vector in the exact training feature order.
2. Replaces missing or invalid values with zero.
3. Scales the feature vector.
4. Predicts the Ridge component.
5. Adds the mean residual MLP correction.
6. Inversely scales the result to degrees Celsius.
7. Returns physical $\Delta T$.

---

## 15. MPC state, action, and prediction variables

### 15.1 State variables

The controller tracks two main physical state variables per building:

$$
x_i(k) = [T_i(k), \mathrm{SOC}_i(k)]^\top
$$

where:

- $T_i(k)$ is the indoor dry-bulb temperature.
- $\mathrm{SOC}_i(k)$ is the electrical battery state of charge.

However, internally, the learned thermal model also uses additional history features:

$$
T_i(k-1), T_i(k-2), T_i(k-3), T_i(k-6), T_i(k-12)
$$

These are stored in `temperature_histories`.

### 15.2 Action variables

For each building, the MPC optimizes two control sequences:

$$
u_i(k) = [u_{\text{bat},i}(k), u_{\text{cool},i}(k)]^\top
$$

where:

- $u_{\text{bat},i}(k)$ controls battery charging and discharging.
- $u_{\text{cool},i}(k)$ controls cooling device power.

Sign convention for battery action:

```text
u_bat > 0  -> battery charging
u_bat < 0  -> battery discharging
u_bat = 0  -> no battery action
```

Cooling action convention:

```text
u_cool = 0  -> cooling device off
u_cool = 1  -> full available cooling power
```

The exact action bounds are not hard-coded. They are extracted from the CityLearn action spaces for each building.

---

## 16. MPC hyperparameters used in the submitted run

The submission notebook configures the MPC as follows:

```python
mpc_config = SIDMPCConfig(
    horizon=4,
    comfort_low=22.0,
    comfort_high=26.0,
    w_track=80.0,
    w_comfort=20.0,
    w_smooth=1.0,
    w_soc=1.0,
    terminal_soc_ref=0.50,
    maxiter=40,
    use_slsqp=True,
    verbose=False,
)
```

### 16.1 Meaning of each parameter

| Parameter | Value | Meaning |
|---|---:|---|
| `horizon` | 4 | Number of future hours optimized at each step |
| `comfort_low` | 22.0 | Lower indoor temperature comfort bound |
| `comfort_high` | 26.0 | Upper indoor temperature comfort bound |
| `w_track` | 80.0 | Weight on local target tracking error |
| `w_comfort` | 20.0 | Weight on comfort violation penalty |
| `w_smooth` | 1.0 | Weight on changes in battery/cooling actions |
| `w_soc` | 1.0 | Weight on terminal SOC regularization |
| `terminal_soc_ref` | 0.50 | Desired terminal battery SOC |
| `maxiter` | 40 | Maximum iterations for the numerical optimizer |
| `use_slsqp` | True | Use SLSQP instead of Powell |
| `verbose` | False | Disable diagnostic printing |

The horizon is intentionally short. The nonlinear optimization is solved separately for 25 buildings at every timestep, so longer horizons increase computational cost substantially.

---

## 17. MPC prediction model

At each predicted step $h$ in the horizon, the controller predicts:

1. next indoor temperature,
2. next battery SOC,
3. local net electricity consumption,
4. comfort violation,
5. tracking error.

### 17.1 Learned thermal prediction

For building $i$ and prediction step $h$:

$$
\widehat{\Delta T}_{i,k+h}
=
f_{\text{SID}}(z_{i,k+h})
$$

Then:

$$
\widehat{T}_{i,k+h+1}
=
\widehat{T}_{i,k+h}
+
\widehat{\Delta T}_{i,k+h}
$$

The feature vector $z_{i,k+h}$ is constructed recursively. Predicted temperatures are fed back into the lag features for future prediction steps.

This means the MPC uses free-running thermal prediction, not just one-step teacher forcing.

### 17.2 Temperature clipping

For numerical stability, predicted temperatures are clipped:

```text
10 °C <= predicted temperature <= 40 °C
```

This does not constrain the actual CityLearn environment. It only prevents the optimizer from producing numerically unrealistic internal predictions.

---

## 18. Future feature construction inside MPC

The MPC must construct the SID feature vector for future steps. The method `_future_base_features` does this.

For a future prediction step $h$, it builds a feature dictionary containing:

- building ID,
- future hour features,
- current or future month features,
- outdoor temperature,
- direct solar irradiance,
- CityLearn-provided 1/2/3-hour forecasts when available,
- fallback future weather arrays for longer horizons,
- current or predicted indoor temperature,
- cooling and heating setpoints,
- comfort band,
- HVAC mode,
- power outage flag,
- battery SOC,
- non-shiftable load,
- DHW demand,
- solar generation,
- candidate cooling action,
- candidate battery action,
- lagged predicted temperatures.

For $h = 1,2,3$, the controller uses the explicit forecast observations when available:

```text
outdoor_dry_bulb_temperature_predicted_1
outdoor_dry_bulb_temperature_predicted_2
outdoor_dry_bulb_temperature_predicted_3
direct_solar_irradiance_predicted_1
direct_solar_irradiance_predicted_2
direct_solar_irradiance_predicted_3
```

For longer horizon steps, it attempts to read future weather arrays from the CityLearn building object.

---

## 19. Battery model

The battery model is approximate and analytical.

The method `_battery_step` estimates the next SOC and battery electricity contribution.

Let:

- $C_i$ be battery capacity,
- $\eta_i$ be round-trip efficiency,
- $\lambda_i$ be loss coefficient,
- $s_i(k)$ be current SOC,
- $a_{\text{bat},i}(k)$ be battery action.

The implementation computes a simplified energy balance:

$$
E_{\text{bal},i}(k) =
a_{\text{bat},i}(k) C_i
$$

Current stored energy after standby loss is:

$$
E_{0,i}(k) =
s_i(k) C_i (1-\lambda_i)
$$

If the battery is charging:

$$
E_{1,i}(k) =
\min(C_i, E_{0,i}(k) + E_{\text{bal},i}(k)\eta_i)
$$

If the battery is discharging:

$$
E_{1,i}(k) =
\max(0, E_{0,i}(k) + \frac{E_{\text{bal},i}(k)}{\eta_i})
$$

The next SOC is:

$$
s_i(k+1) =
\mathrm{clip}\left(\frac{E_{1,i}(k)}{C_i}, 0, 1\right)
$$

The battery electricity contribution used in the net-load estimate is:

$$
P_{\text{bat},i}(k) =
E_{\text{bal},i}(k)
$$

This is approximate because it does not reproduce every internal detail of the CityLearn battery model. It is intended to be simple and computationally cheap inside the MPC loop.

---

## 20. Cooling electricity model

The cooling electricity model is also approximate.

If HVAC mode allows cooling, the cooling electricity is estimated as:

$$
P_{\text{cool},i}(k)
=
u_{\text{cool},i}(k) P_{\text{cool,nom},i}
$$

where:

- $P_{\text{cool,nom},i}$ is the nominal cooling device power,
- $u_{\text{cool},i}(k)$ is the cooling action.

If the HVAC mode does not allow cooling, predicted cooling electricity is set to zero.

If nominal power is unavailable, the controller falls back to a scaled version of the observed cooling electricity consumption.

---

## 21. Approximate net-load balance

The controller estimates local building net electricity as:

$$
\widehat{L}_i(k)
=
L_{\text{base},i}(k)
+
P_{\text{cool},i}(k)
+
P_{\text{bat},i}(k)
$$

where:

$$
L_{\text{base},i}(k)
=
L_{\text{nonshift},i}(k)
+
L_{\text{dhw},i}(k)
+
L_{\text{heating},i}(k)
-
G_{\text{solar},i}(k)
$$

The base net load excludes active cooling and battery control. The MPC then adds its predicted cooling and battery contributions.

This is deliberately simpler than identifying net electricity directly. Net load is largely algebraic in CityLearn, while temperature is the more difficult dynamic process. Therefore, the SID focuses on indoor temperature.

---

## 22. MPC optimization problem

At every timestep and for every building, HRN-MPC solves a finite-horizon optimization problem.

For building $i$ at current time $k$, the decision vector is:

$$
x =
[
u_{\text{bat}}(k),
\ldots,
u_{\text{bat}}(k+H-1),
u_{\text{cool}}(k),
\ldots,
u_{\text{cool}}(k+H-1)
]
$$

The total dimension is:

$$
2H
$$

For the submitted horizon $H = 4$, the optimizer chooses:

```text
4 battery actions + 4 cooling actions = 8 decision variables per building
```

Since there are 25 buildings, each environment step requires solving 25 local optimization problems.

---

## 23. Cost function

The objective combines four terms:

1. tracking error,
2. comfort violation,
3. action smoothness,
4. terminal SOC regularization.

For building $i$, the cost over horizon $H$ is:

$$
J_i =
\sum_{h=0}^{H-1}
\left[
w_{\text{track}}
(\widehat{L}_{i,k+h} - r_{i,k+h})^2
+
w_{\text{comfort}}
(v^{\text{low}}_{i,k+h})^2
+
w_{\text{comfort}}
(v^{\text{high}}_{i,k+h})^2
+
w_{\text{smooth}}
\lVert u_{i,k+h} - u_{i,k+h-1} \rVert_2^2
\right]
+
w_{\text{soc}}
(\widehat{s}_{i,k+H} - s_{\text{ref}})^2
$$

where:

- $\widehat{L}_{i,k+h}$ is predicted local net load,
- $r_{i,k+h}$ is the local target share,
- $v^{\text{low}}$ is lower comfort violation,
- $v^{\text{high}}$ is upper comfort violation,
- $u_{i,k+h}$ is the action vector,
- $\widehat{s}_{i,k+H}$ is terminal SOC,
- $s_{\text{ref}} = 0.5$.

### 23.1 Tracking term

$$
J_{\text{track}}
=
w_{\text{track}}
(\widehat{L}_{i,k+h} - r_{i,k+h})^2
$$

The submitted weight is:

```text
w_track = 80.0
```

This term encourages each building to follow its assigned share of the district target.

### 23.2 Comfort term

The comfort bounds are:

```text
22 °C <= T <= 26 °C
```

Violations are computed as:

$$
v^{\text{low}}_{i,k+h}
=
\max(0, 22 - \widehat{T}_{i,k+h})
$$

$$
v^{\text{high}}_{i,k+h}
=
\max(0, \widehat{T}_{i,k+h} - 26)
$$

The comfort penalty is:

$$
J_{\text{comfort}}
=
w_{\text{comfort}}
\left[
(v^{\text{low}}_{i,k+h})^2
+
(v^{\text{high}}_{i,k+h})^2
\right]
$$

The submitted weight is:

```text
w_comfort = 20.0
```

### 23.3 Action smoothness term

The smoothness penalty is:

$$
J_{\text{smooth}}
=
w_{\text{smooth}}
\left[
(u_{\text{bat},i}(k+h) - u_{\text{bat},i}(k+h-1))^2
+
(u_{\text{cool},i}(k+h) - u_{\text{cool},i}(k+h-1))^2
\right]
$$

The submitted weight is:

```text
w_smooth = 1.0
```

This discourages abrupt switching of battery and cooling actions.

### 23.4 Terminal SOC term

At the end of the prediction horizon, the battery SOC is regularized toward 0.5:

$$
J_{\text{soc}}
=
w_{\text{soc}}
(\widehat{s}_{i,k+H} - 0.5)^2
$$

The submitted weight is:

```text
w_soc = 1.0
```

This discourages the optimizer from always fully charging or fully draining the battery inside the short horizon.

---

## 24. Constraints and bounds

The optimizer uses simple box bounds on the actions.

For each building, the action bounds are extracted from the CityLearn environment:

```python
bat_bounds = action_bounds["electrical_storage"]
cool_bounds = action_bounds["cooling_device"]
```

For all horizon steps:

$$
u_{\text{bat}}^{\min}
\leq
u_{\text{bat}}(k+h)
\leq
u_{\text{bat}}^{\max}
$$

$$
u_{\text{cool}}^{\min}
\leq
u_{\text{cool}}(k+h)
\leq
u_{\text{cool}}^{\max}
$$

Unlike the AL-MPC formulation, this implementation does not explicitly build hard equality constraints for dynamics. Instead, the dynamics are simulated inside the cost function. The optimizer sees a black-box nonlinear objective with action bounds.

Thermal comfort is also treated as a soft penalty rather than a hard constraint. This prevents infeasibility when the comfort band cannot be maintained.

---

## 25. Optimizer

The controller uses SciPy's `minimize`.

The submitted configuration uses:

```text
method = SLSQP
maxiter = 40
ftol = 1e-3
```

If configured with `use_slsqp=False`, the controller can use Powell instead.

### 25.1 Why SLSQP?

SLSQP is suitable here because:

- it supports bounds,
- it can handle nonlinear objectives,
- it is available in SciPy,
- it is easy to integrate in notebooks,
- the horizon is short enough for repeated local optimization.

### 25.2 Why the problem is nonlinear

The MPC objective is nonlinear because the predicted temperature is produced by a Ridge + neural residual model. The prediction is recursively fed back into future features, so later horizon predictions depend nonlinearly on earlier actions.

Therefore, the problem is not a convex quadratic program.

---

## 26. Initial guess

The optimizer initial guess is conservative:

```text
battery action = 0
cooling action = previous cooling action
```

For horizon $H$, the initial decision vector is:

$$
x_0 =
[
0, \ldots, 0,
u_{\text{cool,prev}}, \ldots, u_{\text{cool,prev}}
]
$$

This reduces unnecessary action jumps and gives the optimizer a stable starting point.

---

## 27. Receding-horizon control loop

At each environment timestep:

1. Convert the current observation vector into dictionaries.
2. Update each building's temperature history.
3. For each building:
   - build local target forecast,
   - create future feature vectors,
   - solve the local MPC optimization,
   - extract the first battery and cooling actions.
4. Build CityLearn-compatible action vectors.
5. Apply actions to the environment.
6. Log the new district load.
7. Repeat until the episode terminates.

Only the first action of the optimized sequence is applied. At the next hour, the optimization is solved again using updated observations.

This is the standard receding-horizon MPC principle.

---

## 28. Detailed algorithm

### Offline SID training

```text
1. Create decentralized CityLearn environment.
2. For each rollout:
      a. Reset the environment.
      b. Generate piecewise-random bounded actions.
      c. Step the environment.
      d. Store current observations, actions, and next observations.
3. Add cyclic time features.
4. Add indoor temperature lags: 1, 2, 3, 6, 12 hours.
5. Define target: delta_T = next_T - current_T.
6. Split rollouts into train, validation, and test trajectories.
7. Fit StandardScaler on input features.
8. Fit StandardScaler on delta_T target.
9. Train Ridge NARX model.
10. Compute residual target: true_scaled_delta_T - ridge_prediction.
11. Train 3 residual MLPs with seeds 0, 1, and 2.
12. Evaluate one-step prediction.
13. Evaluate recursive rollout stability.
14. Save scalers, feature list, Ridge model, lags, and MLP weights.
```

### Online MPC control

```text
1. Load CityLearn Texas environment.
2. Load district target for June.
3. Load trained SID artifacts.
4. Configure SIDMPCConfig.
5. Instantiate SIDMPCController.
6. For each simulation hour:
      a. Read current observations.
      b. For each building:
             i. Create local target share.
            ii. Solve nonlinear finite-horizon MPC.
           iii. Return first battery and cooling actions.
      c. Step CityLearn environment.
      d. Log district load and KPIs.
7. Compute final performance metrics.
```

---

## 29. Weights & Biases logging

The submission notebook logs results to W&B:

```text
entity = CityLearn-TeamB
project = CityLearn
run name = MPC_Advanced_SID
```

The logged quantities include:

```text
district_load
NMBE [%]
CV-RMSE [%]
Temp Comfort violation [%]
temperature_plot
```

The district load is logged at each hour using:

```python
wandb.log({
    "hour": t,
    "district_load": float(current_load),
})
```

Final KPIs are logged after the episode:

```python
wandb.log({
    "NMBE [%]": ...,
    "CV-RMSE [%]": ...,
    "Temp Comfort violation [%]": ...,
})
```

A temperature comfort plot is also created using Plotly. The plot includes:

- all building temperatures,
- mean temperature,
- comfort band from 22 °C to 26 °C.

---

## 30. KPI computation

The submission notebook uses the shared KPI function from:

```python
SERVER.KPIs import compute_kpis
```

The KPI inputs are:

```text
district_target_eval
district_load_sid_mpc
sid_mpc_building_temps
```

The main metrics are:

```text
NMBE [%]
CV-RMSE [%]
Temperature Comfort violation [%]
```

NMBE measures average tracking bias:

$$
\mathrm{NMBE}
=
\frac{\mathrm{mean}(y - y_{\text{ref}})}
{\mathrm{mean}(y_{\text{ref}})}
\times 100
$$

CV-RMSE measures overall tracking error:

$$
\mathrm{CVRMSE}
=
\frac{
\sqrt{\mathrm{mean}((y-y_{\text{ref}})^2)}
}
{\mathrm{mean}(y_{\text{ref}})}
\times 100
$$

Comfort violation measures the fraction of indoor temperature samples outside the comfort band:

$$
\mathrm{ComfortViolation}
=
\mathrm{mean}(T < 22 \; \mathrm{or} \; T > 26) \times 100
$$

---

## 31. Technical strengths

### 31.1 More expressive SID than a purely linear model

The residual MLP improves the Ridge NARX baseline by learning nonlinear corrections.

This is useful because building thermal dynamics are affected by nonlinear interactions between:

- outdoor temperature,
- solar irradiance,
- cooling action,
- thermal inertia,
- occupancy-related schedules,
- HVAC operating mode.

### 31.2 Uses persistent excitation

The SID data collection uses randomized actions rather than passive trajectories. This gives the model more information about control response.

### 31.3 Uses recursive rollout evaluation

The notebook explicitly checks whether the model remains stable when recursively feeding its own temperature predictions back into future lag features. This is important for MPC because the controller relies on multi-step predictions.

### 31.4 Keeps battery dynamics analytical

Instead of learning everything, HRN-MPC only learns the difficult part: indoor thermal dynamics. Battery SOC is handled with an analytical approximation.

This makes the control model more structured and interpretable.

### 31.5 Decentralized scalability

Each building solves its own problem. This avoids one very large district-level nonlinear program.

---

## 32. Technical limitations

### 32.1 Equal target split is simple but not optimal

The controller divides the district target equally among 25 buildings. This ignores differences in:

- building size,
- PV generation capacity,
- non-shiftable load,
- battery capacity,
- cooling demand.

A proportional or learned target allocation could improve performance.

### 32.2 Short horizon

The submitted horizon is:

```text
H = 4 hours
```

This keeps runtime manageable but limits look-ahead capability. A longer horizon could improve battery scheduling but would be slower.

### 32.3 Nonlinear optimization is slower than convex MPC

Because the learned model is nonlinear and recursive, the controller uses SLSQP. This is slower and less predictable than solving a convex QP.

### 32.4 Comfort is soft, not guaranteed

Temperature comfort is penalized but not enforced as a hard constraint. Therefore, the optimizer may allow violations if tracking or smoothness terms dominate.

### 32.5 Thermal prediction can drift

Even if one-step prediction is accurate, recursive multi-step prediction can drift. The notebook includes rollout evaluation to diagnose this.

### 32.6 Approximate electricity balance

The net-load calculation uses simplified algebraic approximations for cooling and battery electricity. It may not match CityLearn's exact internal device models in every case.

---

## 33. Recommended future improvements

1. Use proportional target allocation instead of equal target allocation.
2. Increase MPC horizon after runtime optimization.
3. Use direct multi-step SID training rather than only one-step residual learning.
4. Train separate thermal models per building or use learned building embeddings.
5. Add uncertainty-aware MPC using the residual ensemble variance.
6. Add hard or soft SOC constraints closer to CityLearn's exact battery model.
7. Replace SLSQP with a faster optimizer or differentiable MPC formulation.
8. Add warm-starting from the previous optimized sequence.
9. Penalize district-level error directly through a distributed coordination layer.
10. Improve future exogenous feature construction for horizons beyond 3 hours.

---

## 34. Minimal usage example

```python
from citylearn.citylearn import CityLearnEnv
from sid_mpc_controller import LearnedThermalSID, SIDMPCConfig, SIDMPCController

env = CityLearnEnv(
    schema=str(SCHEMA_PATH),
    root_directory=str(DATASET_DIR),
    central_agent=False
)

obs, info = env.reset()

sid_model = LearnedThermalSID(SID_DIR)

config = SIDMPCConfig(
    horizon=4,
    comfort_low=22.0,
    comfort_high=26.0,
    w_track=80.0,
    w_comfort=20.0,
    w_smooth=1.0,
    w_soc=1.0,
    terminal_soc_ref=0.50,
    maxiter=40,
    use_slsqp=True,
)

agent = SIDMPCController(
    env,
    sid_model,
    district_target=district_target,
    config=config
)

while not env.terminated:
    actions = agent.predict(obs)
    obs, _, terminated, truncated, _ = env.step(actions)
    if terminated or truncated:
        break
```

---

## 35. Final conceptual summary

Hybrid Residual-NARX MPC is a data-driven decentralized MPC controller. It first learns a control-oriented thermal model from persistently excited CityLearn rollouts. The learned model combines a linear Ridge NARX baseline with a neural residual ensemble. During control, each building solves a short-horizon nonlinear MPC problem that balances local target tracking, temperature comfort, action smoothness, and terminal battery SOC regularization.

The key idea is:

```text
learn only the difficult thermal dynamics,
approximate the simpler battery/load physics analytically,
then use the learned model inside a receding-horizon optimizer.
```

This makes HRN-MPC a strong intermediate approach between simple rule-based control and fully reinforcement-learning-based control.

