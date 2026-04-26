from typing import Dict, List, Tuple

from sensor_simulator import SensorReading, SensorSimulator
from logic_engine import InferenceResult, LogicEngine
from math_models import MathModels
from optimization import GeneticAlgorithmOptimizer


class GreenhouseAgent:
    """
    Rational Agent for Smart Greenhouse Management.

    Agent cycle:  Perceive → Think → Optimize → Act

    perceive()  : Read sensor data from the environment
    think()     : Apply logic rules and mathematical models
    act()       : Drive actuators based on logical conclusions
    optimize()  : Run Genetic Algorithm to plan the irrigation schedule
    step()      : Execute one full agent cycle
    """

    def __init__(self):
        self.sensor = SensorSimulator()
        self.logic = LogicEngine()
        self.math = MathModels(window_size=30)
        self.optimizer = GeneticAlgorithmOptimizer()

        self.readings_log: List[SensorReading] = []
        self.action_log: List[Dict] = []
        self.current_schedule: List[int] = [0] * 24
        self.actuators: Dict[str, bool] = {
            "irrigation": False,
            "heating": False,
            "cooling": False,
            "ventilation": False,
            "artificial_light": False,
        }

    def perceive(self) -> SensorReading:
        reading = self.sensor.get_reading()
        self.readings_log.append(reading)
        self.math.add_reading(reading)
        return reading

    def think(self, reading: SensorReading) -> Tuple[Dict, List[InferenceResult], Dict]:
        facts, conclusions = self.logic.infer(reading)

        math_metrics = {
            "anomaly_score": self.math.compute_anomaly_score(reading),
            "growth_probability": self.math.compute_growth_probability(reading),
            "expected_yield_30d": self.math.compute_expected_yield(
                self.math.compute_growth_probability(reading)
            ),
            "l2_norm": self.math.compute_l2_norm(reading),
        }

        return facts, conclusions, math_metrics

    def act(self, conclusions: List[InferenceResult], reading: SensorReading):
        for key in self.actuators:
            self.actuators[key] = False

        for conclusion in conclusions:
            action = conclusion.action

            if action == "ACTIVATE_IRRIGATION":
                self.actuators["irrigation"] = True
                self.sensor.irrigate(5.0)
            elif action == "DEACTIVATE_IRRIGATION":
                self.actuators["irrigation"] = False
            elif action == "ACTIVATE_HEATING":
                self.actuators["heating"] = True
                self.sensor.heat(1.0)
            elif action == "ACTIVATE_COOLING":
                self.actuators["cooling"] = True
                self.sensor.cool(1.0)
            elif action == "OPEN_VENTILATION":
                self.actuators["ventilation"] = True
                self.sensor.cool(0.5)
            elif action == "ACTIVATE_ARTIFICIAL_LIGHT":
                self.actuators["artificial_light"] = True

            self.action_log.append({
                "timestamp": reading.timestamp,
                "action": action,
                "reason": conclusion.reason,
                "rule": conclusion.rule,
            })

    def optimize_schedule(self) -> Dict:
        current_soil = self.readings_log[-1].soil_moisture if self.readings_log else 55.0
        best, best_history, avg_history = self.optimizer.optimize(current_soil)
        self.current_schedule = [int(gene > 0.5) for gene in best.chromosome]

        return {
            "schedule": self.current_schedule,
            "fitness": round(best.fitness, 3),
            "best_history": best_history,
            "avg_history": avg_history,
            "total_irrigations": sum(self.current_schedule),
        }

    def step(self) -> Dict:
        reading = self.perceive()
        facts, conclusions, math_metrics = self.think(reading)
        self.act(conclusions, reading)

        return {
            "reading": reading,
            "facts": facts,
            "conclusions": conclusions,
            "math_metrics": math_metrics,
            "actuators": dict(self.actuators),
        }
