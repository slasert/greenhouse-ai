from dataclasses import dataclass
from typing import Dict, List, Tuple
from sensor_simulator import SensorReading


THRESHOLD_TEMP_HIGH = 28.0
THRESHOLD_TEMP_LOW = 15.0
THRESHOLD_HUMIDITY_LOW = 50.0
THRESHOLD_HUMIDITY_HIGH = 85.0
THRESHOLD_SOIL_DRY = 45.0
THRESHOLD_SOIL_WET = 75.0
THRESHOLD_LIGHT_LOW = 3000.0
THRESHOLD_CO2_HIGH = 1200.0
THRESHOLD_CO2_LOW = 600.0


@dataclass
class InferenceResult:
    action: str
    reason: str
    rule: str


class LogicEngine:
    """
    Propositional Logic inference engine.

    Rules (Modus Ponens: P → Q, P ⊢ Q):
        R1: temp_high ∧ humidity_low  → ACTIVATE_COOLING
        R2: temp_high                 → OPEN_VENTILATION
        R3: temp_low                  → ACTIVATE_HEATING
        R4: soil_dry                  → ACTIVATE_IRRIGATION
        R5: soil_wet                  → DEACTIVATE_IRRIGATION
        R6: light_low                 → ACTIVATE_ARTIFICIAL_LIGHT
        R7: co2_high ∨ humidity_high  → OPEN_VENTILATION
        R8: all_normal                → MAINTAIN_STATE
    """

    def evaluate_facts(self, reading: SensorReading) -> Dict[str, bool]:
        t = reading.temperature
        h = reading.humidity
        s = reading.soil_moisture
        l = reading.light_level
        c = reading.co2_level

        return {
            "temp_high":      t > THRESHOLD_TEMP_HIGH,
            "temp_low":       t < THRESHOLD_TEMP_LOW,
            "temp_normal":    THRESHOLD_TEMP_LOW <= t <= THRESHOLD_TEMP_HIGH,
            "humidity_low":   h < THRESHOLD_HUMIDITY_LOW,
            "humidity_high":  h > THRESHOLD_HUMIDITY_HIGH,
            "humidity_normal": THRESHOLD_HUMIDITY_LOW <= h <= THRESHOLD_HUMIDITY_HIGH,
            "soil_dry":       s < THRESHOLD_SOIL_DRY,
            "soil_wet":       s > THRESHOLD_SOIL_WET,
            "soil_normal":    THRESHOLD_SOIL_DRY <= s <= THRESHOLD_SOIL_WET,
            "light_low":      l < THRESHOLD_LIGHT_LOW,
            "co2_high":       c > THRESHOLD_CO2_HIGH,
            "co2_low":        c < THRESHOLD_CO2_LOW,
        }

    def apply_modus_ponens(self, facts: Dict[str, bool]) -> List[InferenceResult]:
        results: List[InferenceResult] = []

        if facts["temp_high"] and facts["humidity_low"]:
            results.append(InferenceResult(
                action="ACTIVATE_COOLING",
                reason=f"Temperature > {THRESHOLD_TEMP_HIGH}°C AND Humidity < {THRESHOLD_HUMIDITY_LOW}%",
                rule="R1: temp_high ∧ humidity_low → ACTIVATE_COOLING",
            ))
        elif facts["temp_high"]:
            results.append(InferenceResult(
                action="OPEN_VENTILATION",
                reason=f"Temperature > {THRESHOLD_TEMP_HIGH}°C",
                rule="R2: temp_high → OPEN_VENTILATION",
            ))

        if facts["temp_low"]:
            results.append(InferenceResult(
                action="ACTIVATE_HEATING",
                reason=f"Temperature < {THRESHOLD_TEMP_LOW}°C",
                rule="R3: temp_low → ACTIVATE_HEATING",
            ))

        if facts["soil_dry"]:
            results.append(InferenceResult(
                action="ACTIVATE_IRRIGATION",
                reason=f"Soil moisture < {THRESHOLD_SOIL_DRY}%",
                rule="R4: soil_dry → ACTIVATE_IRRIGATION",
            ))

        if facts["soil_wet"]:
            results.append(InferenceResult(
                action="DEACTIVATE_IRRIGATION",
                reason=f"Soil moisture > {THRESHOLD_SOIL_WET}%",
                rule="R5: soil_wet → DEACTIVATE_IRRIGATION",
            ))

        if facts["light_low"]:
            results.append(InferenceResult(
                action="ACTIVATE_ARTIFICIAL_LIGHT",
                reason=f"Light level < {THRESHOLD_LIGHT_LOW} lux",
                rule="R6: light_low → ACTIVATE_ARTIFICIAL_LIGHT",
            ))

        if facts["co2_high"] or facts["humidity_high"]:
            results.append(InferenceResult(
                action="OPEN_VENTILATION",
                reason="CO2 level high OR Humidity high",
                rule="R7: co2_high ∨ humidity_high → OPEN_VENTILATION",
            ))

        all_normal = (
            facts["temp_normal"]
            and facts["humidity_normal"]
            and facts["soil_normal"]
            and not facts["co2_high"]
        )
        if all_normal:
            results.append(InferenceResult(
                action="MAINTAIN_STATE",
                reason="All sensor values are within optimal ranges",
                rule="R8: all_normal → MAINTAIN_STATE",
            ))

        if not results:
            results.append(InferenceResult(
                action="MONITOR",
                reason="No active conditions require intervention",
                rule="DEFAULT: → MONITOR",
            ))

        return results

    def infer(self, reading: SensorReading) -> Tuple[Dict[str, bool], List[InferenceResult]]:
        facts = self.evaluate_facts(reading)
        conclusions = self.apply_modus_ponens(facts)
        return facts, conclusions
