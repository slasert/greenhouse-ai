# Artificial Intelligence in Autonomous Environmental Control: A Smart Greenhouse Case Study

---

## Abstract

This essay explores the application of foundational artificial intelligence principles to the design and implementation of an autonomous greenhouse management system. The system acts as a rational agent, integrating propositional logic with Modus Ponens inference, linear algebra-based anomaly detection, probability theory for growth estimation, and a genetic algorithm for irrigation schedule optimization. Each component is examined both theoretically and in the context of its practical application within the system.

---

## 1. Introduction

Artificial intelligence offers powerful tools for automating complex, data-driven decision-making processes. One compelling domain for applying these tools is precision agriculture — specifically, the autonomous control of greenhouse environments. A greenhouse presents a well-defined problem space: a set of measurable environmental variables (temperature, humidity, soil moisture, light intensity, CO2 concentration) must be maintained within optimal ranges to promote plant growth, and actuators (heating, cooling, irrigation, ventilation, artificial lighting) are available to intervene when conditions deviate.

This essay presents a case study in which a rational AI agent is constructed to manage such an environment. The agent's architecture draws on four pillars of classical AI and applied mathematics: propositional logic, linear algebra, probability theory, and evolutionary computation. Together, these tools form a cohesive system capable of perceiving its environment, reasoning about it, and acting effectively under uncertainty.

---

## 2. Rational Agents and the Perceive–Think–Act Cycle

The theoretical backbone of the system is the concept of a **rational agent** as defined by Russell and Norvig (2020): an entity that acts so as to achieve the best expected outcome given its perceptions and knowledge. A rational agent is not simply reactive; it maintains an internal model, applies reasoning, and selects actions that maximize a performance measure.

The greenhouse agent follows a four-phase cycle at each time step:

1. **Perceive** — Sensor readings are collected from the environment: temperature (°C), humidity (%), soil moisture (%), light level (lux), and CO2 concentration (ppm).
2. **Think** — The agent applies propositional logic rules and computes mathematical metrics to assess the current state.
3. **Optimize** — On demand, a genetic algorithm is invoked to plan the next 24-hour irrigation schedule.
4. **Act** — Actuators are driven based on the logical conclusions drawn in the Think phase.

This cycle mirrors the classical agent loop and reflects the design principle that intelligence is not a single computation but an iterative process of sense–reason–respond.

---

## 3. Propositional Logic and Modus Ponens

### 3.1 Propositional Logic

Propositional logic provides a formal language for expressing knowledge as Boolean propositions and reasoning about their truth values. In the greenhouse agent, environmental conditions are encoded as atomic propositions:

- `temp_high`: Temperature > 28°C  
- `soil_dry`: Soil moisture < 40%  
- `co2_high`: CO2 > 1200 ppm  
- `humidity_low`: Humidity < 50%  

These atoms are evaluated from raw sensor readings to produce a **fact base** — a set of propositions known to be true at the current time step.

### 3.2 Modus Ponens Inference

Reasoning proceeds by applying **Modus Ponens**, the classical rule of inference:

```
Premise 1:  P → Q        (If P, then Q)
Premise 2:  P            (P is true)
Conclusion: Q            (Therefore Q)
```

The agent encodes eight production rules of the form `conditions → action`. For example:

| Rule | Antecedent | Consequent |
|------|-----------|-----------|
| R1 | `temp_high ∧ humidity_low` | Activate Cooling |
| R2 | `temp_high` | Open Ventilation |
| R3 | `temp_low` | Activate Heating |
| R4 | `soil_dry` | Activate Irrigation |
| R5 | `soil_wet` | Deactivate Irrigation |
| R6 | `light_low` | Activate Artificial Light |
| R7 | `co2_high ∨ humidity_high` | Open Ventilation |
| R8 | all normal | Maintain Current State |

Each rule's antecedent is evaluated against the fact base. If the antecedent is satisfied, Modus Ponens fires and the consequent action is added to the agent's action set. Multiple rules can fire simultaneously, allowing compound responses (e.g., cooling and artificial lighting activating together).

This rule-based approach provides **transparency** — every action can be traced back to a specific rule and a specific set of sensor readings — which is essential for debugging, validation, and trust in AI-controlled physical systems.

---

## 4. Linear Algebra: Anomaly Detection and Dimensionality Reduction

### 4.1 Sensor State Matrix

At each step, the agent maintains a sliding window of the last *n* sensor readings. These readings are stacked as rows of a **sensor matrix** S ∈ ℝⁿˣ⁵, where each column represents one sensor channel. This matrix is the foundation for all subsequent linear algebra operations.

### 4.2 Covariance Matrix and Eigendecomposition

The **covariance matrix** C ∈ ℝ⁵ˣ⁵ captures how sensor channels vary together:

```
C = (1 / (n-1)) × (S - μ)ᵀ (S - μ)
```

where μ is the row-wise mean (the mean sensor state). Eigendecomposition of C yields eigenvectors (principal directions) and eigenvalues (variance along each direction):

```
C = V Λ Vᵀ
```

The **top eigenvalue λ₁** indicates the dominant source of variation in the sensor space, and the **explained variance ratio** λ₁ / Σλᵢ measures how much of the total variation is captured by the first principal component. This is the essence of **Principal Component Analysis (PCA)**.

### 4.3 Mahalanobis Distance and Anomaly Scoring

The **Mahalanobis distance** measures how far the current sensor reading is from the historical mean, normalized by the covariance structure of the data:

```
d_M(x) = √( (x - μ)ᵀ C⁻¹ (x - μ) )
```

Unlike Euclidean distance, Mahalanobis distance accounts for correlations between variables and differences in scale. A sensor reading that is unusual *relative to the observed distribution* — even if individually within range — will yield a high Mahalanobis distance, triggering an anomaly flag.

### 4.4 L2 Norm

The **L2 norm** of the sensor state vector:

```
‖x‖₂ = √(x₁² + x₂² + x₃² + x₄² + x₅²)
```

provides a simple scalar summary of the overall sensor state magnitude, useful for tracking gross changes in the environment over time.

---

## 5. Probability Theory: Growth Estimation

### 5.1 Gaussian Distributions

For each sensor channel, the agent models the **optimal distribution** as a Gaussian (Normal) distribution centered at the midpoint of the optimal range, with a standard deviation derived from the range width:

```
P(xᵢ) = exp( −(xᵢ − μᵢ)² / (2σᵢ²) )
```

This means readings at the center of the optimal range yield probability close to 1, while readings far from optimal yield probability close to 0 — a smooth, continuous measure of channel health.

### 5.2 Joint Growth Probability

Assuming conditional independence between sensor channels, the **joint growth probability** is the product of individual channel probabilities:

```
P(growth) = P(T) × P(H) × P(S) × P(L) × P(CO₂)
```

This value lies in [0, 1] and reflects the likelihood that current conditions are conducive to healthy plant growth across all dimensions simultaneously.

### 5.3 Expected Yield

The **expected yield** over a 30-day horizon is computed as:

```
E[Yield] = base_yield × P(growth) × 30
```

where `base_yield` is a calibrated per-day yield under ideal conditions. This transforms the abstract probability into a concrete, interpretable forecast (kg / 30 days), directly useful for agricultural planning.

---

## 6. Genetic Algorithm: Irrigation Schedule Optimization

### 6.1 Motivation

While the logic engine handles reactive, real-time control, irrigation planning benefits from a longer-horizon optimization. The question — *which hours of the next 24 should irrigation be active?* — is a combinatorial search problem with 2²⁴ ≈ 16.7 million possible solutions. Exhaustive search is infeasible; a **Genetic Algorithm (GA)** offers an efficient heuristic.

### 6.2 Representation

Each candidate solution is encoded as a **chromosome** of 24 binary genes, where gene *i* = 1 means "irrigate at hour *i*" and gene *i* = 0 means "skip." The population consists of 40 such chromosomes initialized randomly.

### 6.3 Fitness Function

The fitness of a chromosome is evaluated by simulating its effect on soil moisture over 24 hours:

```
f = health_score − water_cost_weight × water_used
```

- **health_score** accumulates a score (0–1) at each hour based on how close the simulated soil moisture stays to the optimal range [50%, 70%].
- **water_cost_weight** penalizes unnecessary irrigation, encoding a trade-off between plant health and water conservation.

### 6.4 Selection, Crossover, and Mutation

**Tournament Selection** (k = 3): Three individuals are drawn at random; the fittest proceeds to reproduction. This preserves selection pressure while maintaining diversity.

**Single-Point Crossover** (rate = 0.8): Two parent chromosomes are split at a random point and recombined. This exchanges large blocks of the irrigation schedule, enabling rapid exploration.

**Bit-Flip Mutation** (rate = 0.05 per gene): Each gene independently has a 5% chance of flipping. Mutation prevents premature convergence and maintains population diversity.

**Elitism**: The best individual found across all generations is always carried into the next generation, guaranteeing that fitness never decreases.

### 6.5 Convergence

Over 60 generations, the population converges toward schedules that maintain soil moisture within the optimal range while minimizing total water usage. The fitness trajectory — best fitness and average fitness per generation — provides a diagnostic view of the algorithm's learning curve.

---

## 7. Integration: The Unified Agent Architecture

The four components do not operate in isolation. They are tightly integrated within the rational agent loop:

- The **sensor simulator** provides the raw perceptual input.
- The **logic engine** uses those inputs to fire rules and select actuator actions.
- The **math models** run in parallel, computing anomaly scores and growth probability from the same sensor data.
- The **genetic algorithm** uses the current soil moisture (from the same perception step) as its initial condition, ensuring that the optimized schedule is grounded in real-time state.
- The **Streamlit dashboard** renders all of this information — gauges, logic conclusions, math metrics, charts, and GA results — in a unified real-time interface.

This architecture demonstrates that classical AI techniques, often taught in isolation, compose naturally into integrated, functional systems.

---

## 8. Conclusion

The smart greenhouse agent illustrates how foundational AI and mathematical tools — propositional logic, linear algebra, probability theory, and evolutionary computation — can be combined to produce a coherent, interpretable, and effective autonomous system. Each technique addresses a distinct aspect of the problem: logic provides explainable rule-based control, linear algebra enables statistical anomaly detection, probability theory quantifies uncertainty in growth outcomes, and the genetic algorithm solves a hard combinatorial planning problem efficiently.

Beyond the specific domain, this project exemplifies a broader principle: intelligence in artificial systems emerges not from any single method, but from the thoughtful integration of multiple complementary approaches, each contributing what it does best.

---

## References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- De Maesschalck, R., Jouan-Rimbaud, D., & Massart, D. L. (2000). The Mahalanobis distance. *Chemometrics and Intelligent Laboratory Systems*, 50(1), 1–18.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
