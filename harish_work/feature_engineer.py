import collections
import numpy as np


class FeatureEngineer:
    """
    Feature engineering helper for the Texas CityLearn dataset.

    This class converts raw CityLearn observations into more useful features
    for rule-based control and later PCA / ICA / entropy analysis.

    Texas action meaning:
    - action[0] = electrical storage / battery
    - action[1] = cooling device / AC
    """

    def __init__(self, observation_names):
        self.names = observation_names

        # Required observation indices
        self.hour_idx = self._get_index("hour")
        self.outdoor_temp_idx = self._get_index("outdoor_dry_bulb_temperature")
        self.net_load_idx = self._get_index("net_electricity_consumption")
        self.solar_gen_idx = self._get_index("solar_generation")
        self.indoor_temp_idx = self._get_index("indoor_dry_bulb_temperature")
        self.cooling_setpoint_idx = self._get_index("indoor_dry_bulb_temperature_cooling_set_point")

        # Dataset is hourly, so 3 samples = 3 hours.
        self.temp_history = collections.deque(maxlen=3)

    def _get_index(self, name):
        """
        Safely find the index of an observation name.
        Returns None if the observation is not available.
        """
        try:
            return self.names.index(name)
        except ValueError:
            print(f"Warning: observation '{name}' not found.")
            return None

    def process_observation(self, obs_array):
        """
        Convert one raw observation array into a dictionary of engineered features.
        """

        # -----------------------------
        # 1. Hour and cyclical time
        # -----------------------------
        hour = self._safe_get(obs_array, self.hour_idx, default=0.0)

        # CityLearn hour is usually 1-24.
        # The sine/cosine encoding makes hour 24 close to hour 1.
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)

        # -----------------------------
        # 2. Outdoor temperature and thermal lag
        # -----------------------------
        outdoor_temp = self._safe_get(obs_array, self.outdoor_temp_idx, default=0.0)

        self.temp_history.append(outdoor_temp)
        rolling_temp_3h = sum(self.temp_history) / len(self.temp_history)

        # -----------------------------
        # 3. True load calculation
        # -----------------------------
        net_load = self._safe_get(obs_array, self.net_load_idx, default=0.0)
        solar_gen = self._safe_get(obs_array, self.solar_gen_idx, default=0.0)

        # CityLearn net load is affected by PV generation.
        # true_load estimates the actual building demand before solar masking.
        true_load = net_load + solar_gen

        # -----------------------------
        # 4. Indoor temperature and cooling setpoint
        # -----------------------------
        indoor_temp = self._safe_get(obs_array, self.indoor_temp_idx, default=0.0)
        cooling_setpoint = self._safe_get(obs_array, self.cooling_setpoint_idx, default=24.0)

        return {
            "hour": hour,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "outdoor_temp": outdoor_temp,
            "rolling_temp_3h": rolling_temp_3h,
            "net_load": net_load,
            "solar_gen": solar_gen,
            "true_load": true_load,
            "indoor_temp": indoor_temp,
            "cooling_setpoint": cooling_setpoint,
        }

    def process_for_ml(self, obs_array):
        """
        Return engineered features as a list.

        This is useful later for PCA / ICA / entropy analysis.
        """
        features = self.process_observation(obs_array)

        return [
            features["sin_hour"],
            features["cos_hour"],
            features["outdoor_temp"],
            features["rolling_temp_3h"],
            features["net_load"],
            features["solar_gen"],
            features["true_load"],
            features["indoor_temp"],
            features["cooling_setpoint"],
        ]

    @staticmethod
    def ml_feature_names():
        """
        Names corresponding to process_for_ml().
        """
        return [
            "sin_hour",
            "cos_hour",
            "outdoor_temp",
            "rolling_temp_3h",
            "net_load",
            "solar_gen",
            "true_load",
            "indoor_temp",
            "cooling_setpoint",
        ]

    @staticmethod
    def _safe_get(obs_array, index, default=0.0):
        """
        Safely read a value from the observation array.
        """
        if index is None:
            return default

        return obs_array[index]


# ==========================================
# Quick standalone test
# ==========================================
if __name__ == "__main__":
    print("Testing FeatureEngineer...")

    fake_names = [
        "hour",
        "outdoor_dry_bulb_temperature",
        "net_electricity_consumption",
        "solar_generation",
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_cooling_set_point",
    ]

    fake_observation = [
        14.0,   # hour
        32.0,   # outdoor temperature
        2.5,    # net electricity consumption
        4.0,    # solar generation
        25.5,   # indoor temperature
        24.0,   # cooling setpoint
    ]

    engineer = FeatureEngineer(fake_names)
    features = engineer.process_observation(fake_observation)

    print("\nEngineered features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")