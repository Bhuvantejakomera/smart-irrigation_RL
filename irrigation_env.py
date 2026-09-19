import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def fao_fixed_reward(growth: float, water_used: float, stress: float) -> float:
    """
    Standard linear reward balancing biomass accumulation, water cost, and stress penalty.
    """
    return 2.0 * growth - 0.15 * water_used - 0.5 * stress


def fao_dynamic_reward(growth: float, water_used: float, day: int, max_days: int, stress: float, moisture: float) -> float:
    """
    FAO-56 Stage-Aware Dynamic Reward function.
    - Dynamically weights crop growth during flowering & yield formation stages.
    - Rewards maintaining soil moisture in optimal Readily Available Water (RAW) range [0.25, 0.38].
    - Penalizes over-irrigation and crop water stress.
    """
    t = day / max_days
    
    # Mid-season stage weight (bell curve peaked around t=0.5, e.g. flowering/grain filling)
    stage_weight = 0.5 + 1.0 * np.exp(-18.0 * (t - 0.5) ** 2)
    
    # Optimal moisture zone bonus (FC = 0.38, RAW threshold = 0.265)
    optimal_moisture_bonus = 0.2 if (0.265 <= moisture <= 0.38) else 0.0
    
    # Instantaneous water productivity (WP) yield per water unit
    wp_bonus = np.tanh(growth / (water_used + 1.0))
    
    reward = (
        stage_weight * growth * 3.0
        + 0.5 * wp_bonus
        + optimal_moisture_bonus
        - 0.12 * water_used
        - 0.8 * stress
    )
    return reward


class FAO56IrrigationEnv(gym.Env):
    """
    FAO-56 Dual Crop Coefficient Hydro-Agronomic Environment for DRL Smart Irrigation.
    
    Soil Physical Parameters:
      - Field Capacity (theta_FC): 0.38 (38% vol)
      - Wilting Point (theta_WP): 0.15 (15% vol)
      - Readily Available Water threshold (theta_CRIT): 0.265 (26.5% vol)
      - Maximum Root Zone Storage depth: 400 mm
      
    Crop Parameters:
      - Baseline ET0: 4.5 mm/day (varying seasonally)
      - Kc_ini = 0.40, Kc_mid = 1.15, Kc_end = 0.55
      
    State Observation Vector (6-D):
      0: Current soil moisture content theta [0.10, 0.45]
      1: Normalized growth stage t in [0.0, 1.0]
      2: Accumulated biomass yield B_t
      3: Daily Crop Evapotranspiration demand ETc (mm)
      4: Crop water stress index Ks in [0.0, 1.0] (1.0 = no stress)
      5: Next day rainfall forecast P_{t+1} (mm)
      
    Action:
      Irrigation volume applied today in range [0, 40] mm/day.
    """
    metadata = {"render_modes": []}

    def __init__(self, reward_type: str = "dynamic"):
        super().__init__()

        self.reward_type = reward_type
        self.max_days = 120
        
        # Soil physics bounds
        self.theta_FC = 0.38
        self.theta_WP = 0.15
        self.theta_CRIT = 0.265  # RAW threshold
        self.root_depth_mm = 400.0  # mm active root zone depth

        self.action_space = spaces.Box(low=0.0, high=40.0, shape=(1,), dtype=np.float32)
        # Observation space bounds
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.5, 1.0, 50.0, 10.0, 1.0, 25.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)

        self.day = 0
        self.soil_moisture = 0.32  # Start at optimal moisture
        self.biomass = 0.1
        
        # Generate weather simulation for 120 days
        days = np.arange(self.max_days)
        # Seasonal ET0 curve (peaks mid-summer)
        self.et0_series = 3.5 + 2.0 * np.sin(np.pi * days / self.max_days) + np.random.normal(0, 0.3, size=self.max_days)
        self.et0_series = np.clip(self.et0_series, 2.0, 7.5)
        
        # Stochastic rainfall pattern (occasional rain events)
        self.rain_series = np.random.exponential(scale=1.2, size=self.max_days)
        # 75% of days have no rain
        self.rain_series[np.random.rand(self.max_days) > 0.25] = 0.0
        self.rain_series = np.clip(self.rain_series, 0.0, 20.0)

        return self._get_state(), {}

    def _get_kc(self, stage: float) -> float:
        """FAO-56 Crop Coefficient curve K_c(t)."""
        if stage < 0.20:
            return 0.40 + (1.15 - 0.40) * (stage / 0.20)
        elif stage < 0.70:
            return 1.15
        else:
            return 1.15 - (1.15 - 0.55) * ((stage - 0.70) / 0.30)

    def step(self, action):
        irrigation = float(np.clip(action[0], 0.0, 40.0))
        
        stage = self.day / self.max_days
        kc = self._get_kc(stage)
        et0 = self.et0_series[self.day]
        etc = kc * et0  # Potential crop evapotranspiration (mm/day)
        
        rainfall = self.rain_series[self.day]
        
        # Soil moisture stress factor Ks (FAO-56)
        if self.soil_moisture >= self.theta_CRIT:
            ks = 1.0
        elif self.soil_moisture <= self.theta_WP:
            ks = 0.0
        else:
            ks = (self.soil_moisture - self.theta_WP) / (self.theta_CRIT - self.theta_WP + 1e-6)
            
        # Actual evapotranspiration ETa
        eta = ks * etc
        
        # Soil water balance update in depth (mm)
        current_water_mm = self.soil_moisture * self.root_depth_mm
        added_water_mm = irrigation + rainfall
        removed_water_mm = eta
        
        new_water_mm = current_water_mm + added_water_mm - removed_water_mm
        max_water_mm = self.theta_FC * self.root_depth_mm
        
        # Runoff & Deep Percolation when exceeding Field Capacity
        runoff_mm = max(0.0, new_water_mm - max_water_mm)
        new_water_mm = min(new_water_mm, max_water_mm)
        
        self.soil_moisture = float(np.clip(new_water_mm / self.root_depth_mm, self.theta_WP * 0.8, 0.45))
        
        # Biomass growth model: proportional to actual ETa and crop developmental stage factor
        stage_growth_potential = 0.04 + 0.06 * np.sin(np.pi * stage)
        growth = (eta / 5.0) * stage_growth_potential
        self.biomass += growth
        
        stress_penalty = 1.0 - ks
        
        if self.reward_type == "fixed":
            reward = fao_fixed_reward(growth, irrigation, stress_penalty)
        else:
            reward = fao_dynamic_reward(growth, irrigation, self.day, self.max_days, stress_penalty, self.soil_moisture)

        self.day += 1
        terminated = self.day >= self.max_days
        truncated = False

        return self._get_state(), reward, terminated, truncated, {
            "irrigation": irrigation,
            "rainfall": rainfall,
            "etc": etc,
            "eta": eta,
            "ks": ks,
            "runoff": runoff_mm,
            "biomass_gain": growth,
        }

    def _get_state(self):
        stage = min(1.0, self.day / self.max_days)
        kc = self._get_kc(stage)
        et0 = self.et0_series[min(self.day, self.max_days - 1)]
        etc = kc * et0
        
        # Soil stress factor Ks
        if self.soil_moisture >= self.theta_CRIT:
            ks = 1.0
        elif self.soil_moisture <= self.theta_WP:
            ks = 0.0
        else:
            ks = (self.soil_moisture - self.theta_WP) / (self.theta_CRIT - self.theta_WP + 1e-6)
            
        next_day = min(self.day + 1, self.max_days - 1)
        rain_forecast = self.rain_series[next_day]

        return np.array([
            self.soil_moisture,
            stage,
            self.biomass,
            etc,
            ks,
            rain_forecast,
        ], dtype=np.float32)


class RuleBasedAgronomicAgent:
    """
    Traditional Rule-Based Agricultural Controller (Baseline).
    Triggers irrigation whenever soil moisture drops below Readily Available Water (RAW) threshold.
    """
    def __init__(self, theta_crit=0.265, theta_target=0.36, root_depth=400.0):
        self.theta_crit = theta_crit
        self.theta_target = theta_target
        self.root_depth = root_depth

    def predict(self, obs, deterministic=True):
        soil_moisture = obs[0]
        rain_forecast = obs[5]
        
        if soil_moisture < self.theta_crit:
            # Calculate water deficit up to target field capacity, subtracting expected rainfall
            water_deficit_mm = (self.theta_target - soil_moisture) * self.root_depth
            irrigation = max(0.0, water_deficit_mm - rain_forecast)
            irrigation = min(35.0, irrigation)
        else:
            irrigation = 0.0
            
        return np.array([irrigation], dtype=np.float32), None


def make_env(reward_type: str, seed: int):
    def _init():
        env = FAO56IrrigationEnv(reward_type=reward_type)
        env.reset(seed=seed)
        return env
    return _init


def get_env(reward_type: str, n_envs: int = 4):
    return DummyVecEnv([make_env(reward_type, i) for i in range(n_envs)])
