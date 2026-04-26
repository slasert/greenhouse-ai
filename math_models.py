import numpy as np
from typing import Dict, List, Tuple
from sensor_simulator import SensorReading


OPTIMAL_TEMP = (22.0, 3.0)
OPTIMAL_HUMIDITY = (70.0, 10.0)
OPTIMAL_SOIL = (60.0, 8.0)
OPTIMAL_LIGHT = (6000.0, 1000.0)
OPTIMAL_CO2 = (1000.0, 150.0)

MAX_YIELD_PER_DAY = 0.5


class MathModels:
    """
    Mathematical foundation of the greenhouse agent.

    Linear Algebra:
        - Sensor matrix X  (n_readings × 5)
        - Covariance matrix C = cov(X)
        - Eigendecomposition: C · v = λ · v  (PCA)
        - Mahalanobis distance: d = sqrt((x-μ)ᵀ C⁻¹ (x-μ))
        - L2 norm: ||x||₂

    Probability:
        - Gaussian likelihood per sensor channel
        - Joint growth probability (independent assumption)
        - Expected yield: E[Y] = p_growth · max_yield · days
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history: List[SensorReading] = []

    def add_reading(self, reading: SensorReading):
        self.history.append(reading)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_sensor_matrix(self) -> np.ndarray:
        if not self.history:
            return np.zeros((1, 5))
        return np.array([
            [
                r.temperature,
                r.humidity,
                r.soil_moisture,
                r.light_level / 100.0,
                r.co2_level / 10.0,
            ]
            for r in self.history
        ])

    def compute_covariance_matrix(self) -> np.ndarray:
        X = self.get_sensor_matrix()
        if X.shape[0] < 2:
            return np.eye(5)
        return np.cov(X.T)

    def compute_principal_components(self) -> Tuple[np.ndarray, np.ndarray]:
        cov = self.compute_covariance_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvalues[order], eigenvectors[:, order]

    def compute_anomaly_score(self, reading: SensorReading) -> float:
        X = self.get_sensor_matrix()
        if X.shape[0] < 3:
            return 0.0

        x = np.array([
            reading.temperature,
            reading.humidity,
            reading.soil_moisture,
            reading.light_level / 100.0,
            reading.co2_level / 10.0,
        ])
        mean = X.mean(axis=0)
        cov = np.cov(X.T) + np.eye(5) * 1e-6

        diff = x - mean
        try:
            cov_inv = np.linalg.inv(cov)
            score = float(np.sqrt(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            score = float(np.linalg.norm(diff))

        return round(score, 3)

    def compute_l2_norm(self, reading: SensorReading) -> float:
        x = np.array([
            reading.temperature / 40.0,
            reading.humidity / 100.0,
            reading.soil_moisture / 100.0,
            reading.light_level / 10000.0,
            reading.co2_level / 2000.0,
        ])
        return round(float(np.linalg.norm(x)), 4)

    def compute_growth_probability(self, reading: SensorReading) -> float:
        params = [
            (reading.temperature, *OPTIMAL_TEMP),
            (reading.humidity, *OPTIMAL_HUMIDITY),
            (reading.soil_moisture, *OPTIMAL_SOIL),
            (reading.light_level, *OPTIMAL_LIGHT),
            (reading.co2_level, *OPTIMAL_CO2),
        ]
        prob = 1.0
        for val, mu, sigma in params:
            p = np.exp(-0.5 * ((val - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            p_max = 1.0 / (sigma * np.sqrt(2 * np.pi))
            prob *= p / p_max

        return round(float(np.clip(prob, 0.0, 1.0)), 4)

    def compute_expected_yield(self, growth_probability: float, days: int = 30) -> float:
        return round(growth_probability * MAX_YIELD_PER_DAY * days, 3)

    def get_statistics(self) -> Dict:
        if not self.history:
            return {}
        X = self.get_sensor_matrix()
        eigenvalues, eigenvectors = self.compute_principal_components()
        total_variance = eigenvalues.sum()
        return {
            "sensor_matrix_shape": X.shape,
            "means": X.mean(axis=0).round(3).tolist(),
            "stds": X.std(axis=0).round(3).tolist(),
            "top_eigenvalue": round(float(eigenvalues[0]), 3),
            "top_eigenvector": eigenvectors[:, 0].round(3).tolist(),
            "explained_variance_ratio": round(float(eigenvalues[0] / total_variance), 3) if total_variance > 0 else 0.0,
        }
