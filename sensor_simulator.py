import numpy as np
from dataclasses import dataclass


TEMP_BASE = 22.0
HUMIDITY_BASE = 65.0
SOIL_BASE = 60.0
LIGHT_BASE = 6000.0
CO2_BASE = 900.0


@dataclass
class SensorReading:
    temperature: float
    humidity: float
    soil_moisture: float
    light_level: float
    co2_level: float
    timestamp: float


class SensorSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.base_temp = TEMP_BASE
        self.base_humidity = HUMIDITY_BASE
        self.base_soil = SOIL_BASE
        self.base_light = LIGHT_BASE
        self.base_co2 = CO2_BASE

    def get_reading(self) -> SensorReading:
        self.time += 1
        hour = (self.time / 60) % 24

        temp = self.base_temp + 5 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 0.5)

        humidity = self.base_humidity - 0.5 * (temp - self.base_temp) + self.rng.normal(0, 1.0)
        humidity = float(np.clip(humidity, 0, 100))

        self.base_soil = max(20.0, self.base_soil - 0.05 + self.rng.normal(0, 0.1))
        soil = float(np.clip(self.base_soil + self.rng.normal(0, 0.5), 0, 100))

        light = float(max(0.0, self.base_light * np.sin(np.pi * hour / 12) + self.rng.normal(0, 200)))

        co2 = float(np.clip(self.base_co2 + self.rng.normal(0, 20), 400, 2000))

        return SensorReading(
            temperature=round(float(temp), 2),
            humidity=round(humidity, 2),
            soil_moisture=round(soil, 2),
            light_level=round(light, 2),
            co2_level=round(co2, 2),
            timestamp=float(self.time),
        )

    def irrigate(self, amount: float = 10.0):
        self.base_soil = min(100.0, self.base_soil + amount)

    def heat(self, amount: float = 2.0):
        self.base_temp = min(40.0, self.base_temp + amount)

    def cool(self, amount: float = 2.0):
        self.base_temp = max(0.0, self.base_temp - amount)
