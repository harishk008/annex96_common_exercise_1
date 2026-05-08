Texas CityLearn ML Feature Dataset
==================================

Purpose
-------
This folder contains ML-ready feature datasets generated from the cleaned Texas
CityLearn / Annex 96 baseline simulation.

The goal is to provide enriched datasets that can be used by the team to build:
- reinforcement learning models,
- supervised machine learning models,
- forecasting models,
- feature-importance analysis,
- target-tracking analysis.

The generated datasets preserve the original physical features and add new
time-series, statistical, PCA, ICA, and entropy-based features.


Input Files
-----------
The feature generation script reads:

1. tx_real_baseline_building_features.csv
   - Building-level baseline dataset.
   - One row represents one building at one hourly time step.
   - Expected size: 25 buildings x 168 hours = 4200 rows.

2. tx_real_baseline_portfolio_features.csv
   - Portfolio-level baseline dataset.
   - One row represents the full portfolio at one hourly time step.
   - Expected size: 168 rows.


Output Files
------------
The generated ML-ready files are:

1. tx_building_ml_features.csv
   - Building-level ML dataset.
   - Useful for building-level ML models and local controller learning.
   - Each row represents one building at one hourly time step.

2. tx_portfolio_ml_features.csv
   - Portfolio-level ML dataset.
   - Useful for portfolio-level target tracking, forecasting, and RL training.
   - Each row represents one hourly portfolio state.

3. tx_feature_metadata.json
   - Machine-readable description of the generated datasets.

4. README_feature_generation.txt
   - This explanation file.


Feature Groups
--------------
The generated datasets include the following groups of features.

1. Original Physical Features
-----------------------------
These are the main physical signals from the CityLearn simulation:

- net_load
- solar_gen
- true_load
- indoor_temp
- outdoor_temp
- cooling_setpoint
- portfolio_net_load
- portfolio_solar_generation
- portfolio_true_load
- district_target

These are kept because they are directly meaningful for energy control.

2. Controller Action Features
-----------------------------
The baseline controller actions are included:

- battery_action
- cooling_action
- average_battery_action
- average_cooling_action

These allow the ML/RL model to understand what action was applied by the
rule-based baseline controller.

3. Time Features
----------------
The following time-related features are included:

- hour
- hour_of_day
- sin_hour
- cos_hour
- is_solar_hour
- is_evening_peak
- is_night

The sine and cosine hour features encode the daily cycle more correctly than
plain hour values. For example, hour 23 and hour 0 are close in real time, and
sin/cos encoding helps a model understand this cyclic relationship.

4. Tracking Error Features
--------------------------
Portfolio-level tracking error is:

    tracking_error = portfolio_net_load - district_target

Building-level tracking error proxy is:

    tracking_error_proxy = net_load - district_target / number_of_buildings

The building-level version is only a proxy because the district target is defined
for the full portfolio, not for individual buildings.

5. Lag Features
---------------
Lag features store previous signal values:

- _lag_1
- _lag_2
- _lag_3

For hourly data, these correspond to the previous 1, 2, and 3 hours.

Examples:

- net_load_lag_1
- solar_gen_lag_1
- tracking_error_lag_1
- portfolio_net_load_lag_1
- district_target_lag_1

These features help ML models understand short-term memory and trends.

6. Ramp Features
----------------
Ramp features measure how quickly a signal changes:

    ramp = current value - previous value

Examples:

- net_load_ramp
- true_load_ramp
- solar_ramp
- portfolio_net_load_ramp
- district_target_ramp
- tracking_error_ramp

These are useful because a controller must respond not only to the current load,
but also to how quickly the load or target is changing.

7. Rolling Window Features
--------------------------
Rolling features summarize recent history over short time windows:

- 3-hour rolling mean
- 6-hour rolling mean
- 24-hour rolling mean
- rolling standard deviation

Examples:

- net_load_rolling_mean_3h
- net_load_rolling_std_3h
- tracking_error_rolling_mean_3h
- tracking_error_rolling_std_3h

Rolling means show recent average behavior.
Rolling standard deviations show short-term variability.

8. Entropy Features
-------------------
Entropy features measure how variable or unpredictable a signal is over a
24-hour rolling window.

Examples:

- net_load_entropy_24h
- solar_gen_entropy_24h
- tracking_error_entropy_24h
- portfolio_net_load_entropy_24h

Higher entropy means the signal changes in a less predictable way. This may be
useful for RL models because periods with high entropy may require more adaptive
control.

9. PCA Features
---------------
PCA stands for Principal Component Analysis.

PCA creates compressed components from correlated numerical features. The PCA
features are:

- building_pca_1
- building_pca_2
- building_pca_3
- portfolio_pca_1
- portfolio_pca_2
- portfolio_pca_3

These components can reduce redundancy between features such as temperature,
cooling action, load, solar generation, and tracking error.

10. ICA Features
----------------
ICA stands for Independent Component Analysis.

ICA attempts to separate hidden independent patterns from the feature set. The
ICA features are:

- building_ica_1
- building_ica_2
- building_ica_3
- portfolio_ica_1
- portfolio_ica_2
- portfolio_ica_3

These may help separate hidden effects such as solar-driven behavior,
temperature-driven behavior, load-driven behavior, and control-action behavior.


How to Use These Datasets
-------------------------

For portfolio-level RL or ML:
Use:

    tx_portfolio_ml_features.csv

Recommended target/output variables:
- tracking_error
- portfolio_net_load
- district_target
- average_battery_action
- average_cooling_action

For building-level ML:
Use:

    tx_building_ml_features.csv

Useful for:
- building behavior modeling,
- local load prediction,
- clustering buildings,
- studying action-response behavior.

Important Notes
---------------
1. The dataset uses hourly time steps.
2. The selected evaluation window is the first active district target period.
3. The Texas dataset uses the following active actions:
   - electrical_storage
   - cooling_device
4. The Vermont dataset is not used in this feature generation process.
5. Original physical features are not removed.
6. PCA and ICA features are additional features, not replacements.
7. Entropy is calculated using a 24-hour rolling window.
8. The building-level tracking error is a proxy because the target is defined at
   the portfolio level.


Suggested Team Usage
--------------------
The team can test different feature sets:

1. Physical features only
2. Physical + lag/ramp features
3. Physical + rolling/entropy features
4. Physical + PCA/ICA features
5. All features together

This allows comparison of whether PCA, ICA, and entropy improve ML/RL model
performance.


Generated Files Location
------------------------
All generated files are saved in:

    /Users/harish/Desktop/project/Team-internship-project/annex96_common_exercise_1/harish_work/outputs/ml_datasets