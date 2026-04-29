#!/usr/bin/env python3
"""
Gymnasium environment for PPO reinforcement learning agent.
State: 5 sensors + 5 actuators + hour_of_day (11-dim)
Action: Discrete 32 (5-bit binary actuator combinations)
Reward: +P(growth) - energy_cost - critical_penalties
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

SENSOR_COLS = ["temperature_c", "humidity_pct", "soil_moisture_pct",
               "light_lux", "co2_ppm"]

OPTIMAL = {
    "temperature_c":     (20.0, 26.0),
    "humidity_pct":      (60.0, 80.0),
    "soil_moisture_pct": (50.0, 75.0),
    "light_lux":         (20000, 60000),
    "co2_ppm":           (400.0, 800.0),
}
NORM = {
    "temperature_c":     (15, 35),
    "humidity_pct":      (20, 95),
    "soil_moisture_pct": (10, 90),
    "light_lux":         (0, 85000),
    "co2_ppm":           (350, 1500),
}


class GreenhouseEnv(gym.Env):
    """Greenhouse control environment for PPO training."""
    metadata = {"render_modes": []}

    def __init__(self, data_csv: Optional[Path] = None, max_steps: int = 96) -> None:
        super().__init__()
        self.max_steps = max_steps

        if data_csv and Path(data_csv).exists():
            df = pd.read_csv(data_csv)
            self.sensor_data = df[SENSOR_COLS].values.astype(np.float32)
        else:
            self.sensor_data = None

        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(11,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(32)

        self.sensor_state = np.zeros(5, dtype=np.float32)
        self.actuator_state = np.zeros(5, dtype=np.int32)
        self.step_count = 0
        self.data_idx = 0

    def _normalize_sensors(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros(5, dtype=np.float32)
        for i, col in enumerate(SENSOR_COLS):
            lo, hi = NORM[col]
            out[i] = (s[i] - lo) / (hi - lo + 1e-8) * 2 - 1
        return np.clip(out, -3, 3)

    def _compute_pgrowth(self, s: np.ndarray) -> float:
        """Product of per-sensor optimality scores."""
        score = 1.0
        for i, col in enumerate(SENSOR_COLS):
            lo, hi = OPTIMAL[col]
            mid = (lo + hi) / 2
            width = (hi - lo) / 2
            score *= np.exp(-0.5 * ((s[i] - mid) / (width + 1e-8)) ** 2)
        return float(np.clip(score, 0, 1))

    def _action_to_actuators(self, action: int) -> np.ndarray:
        """Decode integer action (0-31) to 5-bit binary vector."""
        return np.array([(action >> i) & 1 for i in range(5)], dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        norm_sensors = self._normalize_sensors(self.sensor_state)
        hour_feat = np.array([(self.step_count % 96) / 96.0], dtype=np.float32)
        act_feat = self.actuator_state.astype(np.float32)
        return np.concatenate([norm_sensors, act_feat, hour_feat])

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.sensor_data is not None:
            self.data_idx = self.np_random.integers(0, len(self.sensor_data))
            self.sensor_state = self.sensor_data[self.data_idx].copy()
        else:
            self.sensor_state = np.array([22.0, 65.0, 60.0, 30000.0, 600.0], dtype=np.float32)
        self.actuator_state = np.zeros(5, dtype=np.int32)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.actuator_state = self._action_to_actuators(action)

        s = self.sensor_state.copy()
        s[0] += 0.5 * self.actuator_state[1] - 0.5 * self.actuator_state[2]
        s[1] -= 2.0 * self.actuator_state[3] - 3.0 * self.actuator_state[0]
        s[2] += 5.0 * self.actuator_state[0] - 0.3
        s[3] += 5000 * self.actuator_state[4]
        s[4] -= 30.0 * self.actuator_state[3]

        s[0] += 0.1 * (20.0 - s[0]) + self.np_random.normal(0, 0.3)
        s[1] += self.np_random.normal(0, 0.5)
        s[2] = max(10.0, s[2] - 0.2)
        s[4] += self.np_random.normal(0, 10)

        self.sensor_state = np.clip(s, [0, 0, 0, 0, 350],
                                    [45, 100, 100, 90000, 2000]).astype(np.float32)

        pgrowth = self._compute_pgrowth(self.sensor_state)
        energy_cost = 0.05 * int(self.actuator_state.sum())
        critical_pen = -1.0 if (self.sensor_state[0] > 40 or
                                self.sensor_state[0] < 5 or
                                self.sensor_state[2] < 15) else 0.0
        reward = pgrowth - energy_cost + critical_pen

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        info = {"pgrowth": pgrowth, "energy_cost": energy_cost}

        return self._get_obs(), reward, terminated, False, info
