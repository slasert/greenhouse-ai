import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


CHROMOSOME_LENGTH = 24
IRRIGATE_AMOUNT = 5.0
EVAPORATION_PER_HOUR = 0.8
SOIL_OPTIMAL_MIN = 50.0
SOIL_OPTIMAL_MAX = 70.0
WATER_COST_WEIGHT = 0.5


@dataclass
class Individual:
    chromosome: np.ndarray
    fitness: float = 0.0


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for optimizing the 24-hour irrigation schedule.

    Encoding:
        Chromosome  : binary array of length 24 (gene i = 1 → irrigate at hour i)

    Operators:
        Selection   : Tournament selection (k = 3)
        Crossover   : Single-point crossover
        Mutation    : Bit-flip mutation

    Fitness:
        f = health_score - water_cost_weight × water_used
        health_score accumulates how close soil moisture stays to [50, 70] %
    """

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 60,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def _initialize_population(self) -> List[Individual]:
        return [
            Individual(chromosome=self.rng.integers(0, 2, CHROMOSOME_LENGTH).astype(float))
            for _ in range(self.population_size)
        ]

    def _evaluate_fitness(self, chromosome: np.ndarray, initial_soil: float) -> float:
        soil = initial_soil
        health_score = 0.0
        water_used = 0.0

        for hour in range(CHROMOSOME_LENGTH):
            soil = max(0.0, soil - EVAPORATION_PER_HOUR)
            if chromosome[hour] > 0.5:
                soil = min(100.0, soil + IRRIGATE_AMOUNT)
                water_used += 1.0

            if SOIL_OPTIMAL_MIN <= soil <= SOIL_OPTIMAL_MAX:
                health_score += 1.0
            elif soil < SOIL_OPTIMAL_MIN:
                health_score += max(0.0, 1.0 - (SOIL_OPTIMAL_MIN - soil) / SOIL_OPTIMAL_MIN)
            else:
                health_score += max(0.0, 1.0 - (soil - SOIL_OPTIMAL_MAX) / (100.0 - SOIL_OPTIMAL_MAX))

        return health_score - WATER_COST_WEIGHT * water_used

    def _tournament_selection(self, population: List[Individual], k: int = 3) -> Individual:
        indices = self.rng.choice(len(population), k, replace=False)
        return max((population[i] for i in indices), key=lambda ind: ind.fitness)

    def _crossover(self, parent_a: Individual, parent_b: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return Individual(parent_a.chromosome.copy()), Individual(parent_b.chromosome.copy())
        point = int(self.rng.integers(1, CHROMOSOME_LENGTH - 1))
        child_a = np.concatenate([parent_a.chromosome[:point], parent_b.chromosome[point:]])
        child_b = np.concatenate([parent_b.chromosome[:point], parent_a.chromosome[point:]])
        return Individual(child_a), Individual(child_b)

    def _mutate(self, individual: Individual) -> Individual:
        chromosome = individual.chromosome.copy()
        for i in range(len(chromosome)):
            if self.rng.random() < self.mutation_rate:
                chromosome[i] = 1.0 - chromosome[i]
        return Individual(chromosome)

    def optimize(self, initial_soil: float = 55.0) -> Tuple[Individual, List[float], List[float]]:
        population = self._initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []

        for ind in population:
            ind.fitness = self._evaluate_fitness(ind.chromosome, initial_soil)

        best_ever = max(population, key=lambda ind: ind.fitness)
        best_ever = Individual(best_ever.chromosome.copy(), best_ever.fitness)

        for _ in range(self.generations):
            next_generation: List[Individual] = [
                Individual(best_ever.chromosome.copy(), best_ever.fitness)
            ]

            while len(next_generation) < self.population_size:
                parent_a = self._tournament_selection(population)
                parent_b = self._tournament_selection(population)
                child_a, child_b = self._crossover(parent_a, parent_b)
                child_a = self._mutate(child_a)
                child_b = self._mutate(child_b)
                child_a.fitness = self._evaluate_fitness(child_a.chromosome, initial_soil)
                child_b.fitness = self._evaluate_fitness(child_b.chromosome, initial_soil)
                next_generation.extend([child_a, child_b])

            population = next_generation[: self.population_size]

            generation_best = max(population, key=lambda ind: ind.fitness)
            if generation_best.fitness > best_ever.fitness:
                best_ever = Individual(generation_best.chromosome.copy(), generation_best.fitness)

            self.best_fitness_history.append(best_ever.fitness)
            self.avg_fitness_history.append(float(np.mean([ind.fitness for ind in population])))

        return best_ever, self.best_fitness_history, self.avg_fitness_history
