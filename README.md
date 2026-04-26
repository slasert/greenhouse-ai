# 🌿 Smart Green House Artificial Intelligence

> An autonomous rational agent for real-time greenhouse environment management — built as a final project for the **Principles of Artificial Intelligence** course.

---

## Overview

This project implements a **rational AI agent** that continuously monitors a simulated greenhouse environment and makes intelligent decisions to maintain optimal growing conditions. The agent integrates four core AI and mathematics concepts into a unified, real-time dashboard.

---

## AI & Mathematical Foundations

### 🧠 Propositional Logic — Modus Ponens
The agent's reasoning engine encodes **8 propositional rules** evaluated at every step. Logical inference is performed using the **Modus Ponens** rule of inference:

```
P → Q
P
─────
∴ Q
```

Example rules:
- `High Temperature ∧ Low Humidity → Activate Cooling`
- `Dry Soil → Activate Irrigation`
- `All Normal → Maintain Current State`

### 📐 Linear Algebra & PCA
Sensor readings are organized into a **state matrix** and processed using:
- **Covariance Matrix** computation from the sensor window
- **Eigendecomposition** to extract principal components (PCA)
- **Mahalanobis Distance** for anomaly scoring
- **L2 Norm** of the full sensor state vector

### 📊 Probability Theory
Growth probability is estimated using **Gaussian (Normal) distributions** for each sensor channel. A **joint probability** is computed across all channels and mapped to an expected yield:

```
P(growth) = P(T_opt) × P(H_opt) × P(S_opt) × P(L_opt) × P(CO2_opt)
E[Yield] = base_yield × P(growth) × 30
```

### 🧬 Genetic Algorithm — Irrigation Optimizer
A **Genetic Algorithm** optimizes the 24-hour irrigation schedule:

| Parameter   | Value |
|-------------|-------|
| Chromosome  | 24 binary genes (1 = irrigate at hour h) |
| Population  | 40 individuals |
| Generations | 60 |
| Selection   | Tournament (k = 3) |
| Crossover   | Single-point (rate = 0.8) |
| Mutation    | Bit-flip (rate = 0.05) |
| Elitism     | Best individual always preserved |

Fitness function:
```
f = health_score − water_cost_weight × water_used
```

---

## Features

- 📡 **Real-time sensor simulation** — Temperature, Humidity, Soil Moisture, Light, CO2
- 🧠 **Logic engine** — 8 propositional rules, Modus Ponens inference
- 📐 **Math models** — PCA, Mahalanobis anomaly detection, L2 norm
- 📊 **Growth probability** — Gaussian joint probability, 30-day yield forecast
- 🧬 **GA optimizer** — Evolves optimal 24-hour irrigation schedules
- ⚙️ **Actuator control** — Irrigation, Heating, Cooling, Ventilation, Artificial Light
- 📈 **Live charts** — Sensor history, fitness convergence, irrigation schedule
- 🎛️ **Manual overrides** — Force irrigate, heat, or cool at any step

---

## Project Structure

```
greenhouse_ai/
├── main.py               # Streamlit UI & dashboard
├── agent.py              # Rational agent (Perceive → Think → Optimize → Act)
├── sensor_simulator.py   # Simulated sensor environment
├── logic_engine.py       # Propositional logic & Modus Ponens
├── math_models.py        # Linear algebra, PCA, probability models
├── optimization.py       # Genetic Algorithm optimizer
├── requirements.txt      # Python dependencies
└── images/               # Background photos for UI
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Streamlit | Interactive web dashboard |
| NumPy | Matrix operations, PCA, eigendecomposition |
| Pandas | Sensor data logging & display |
| Plotly | Gauge charts, time-series, GA fitness plots |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/greenhouse-ai.git
cd greenhouse-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

Then open **http://localhost:8501** in your browser.

---

## Agent Cycle

```
┌─────────────┐
│   PERCEIVE  │  ← Read 5 sensor channels
└──────┬──────┘
       ↓
┌─────────────┐
│    THINK    │  ← Apply logic rules + compute math metrics
└──────┬──────┘
       ↓
┌─────────────┐
│   OPTIMIZE  │  ← Run Genetic Algorithm (on demand)
└──────┬──────┘
       ↓
┌─────────────┐
│     ACT     │  ← Drive actuators based on conclusions
└─────────────┘
```

---

## Course

**Principles of Artificial Intelligence** — Final Project  
Topics covered: Rational Agents, Propositional Logic, Search & Optimization, Probability Theory, Linear Algebra in AI

---

*Built with 🌿 and Python*
