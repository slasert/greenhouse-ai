#!/usr/bin/env python3
"""
UCI Soil Sensor Dataset Loader & Preprocessor
Generates synthetic greenhouse data + optional real UCI data.
Outputs: datasets/processed/sensors_{train,val,test}.csv + scaler.pkl
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

SENSOR_COLS = ["temperature_c", "humidity_pct", "soil_moisture_pct",
               "light_lux", "co2_ppm"]


def generate_synthetic_greenhouse(n_rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic greenhouse sensor data with realistic dynamics."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    hour = (t // 4) % 24  # 15-min intervals

    # Diurnal temperature: 18–28°C peak at 14:00
    temp = 23 + 5 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 0.8, n_rows)

    # Humidity: inversely correlated with temp, 40–80%
    humidity = 65 - 0.8 * (temp - 23) + rng.normal(0, 2.0, n_rows)
    humidity = np.clip(humidity, 30, 95)

    # Soil moisture: drops between irrigation events, 30–80%
    soil = np.zeros(n_rows)
    soil[0] = 60.0
    for i in range(1, n_rows):
        drain = 0.05 + 0.02 * rng.random()
        irrigate = 15.0 if (i % 96 == 0) else 0.0
        soil[i] = np.clip(soil[i-1] - drain + irrigate + rng.normal(0, 0.3), 20, 90)

    # Light: solar curve 0–80,000 lux
    light = np.maximum(0, 40000 * np.sin((hour - 6) * np.pi / 12))
    light += rng.normal(0, 1000, n_rows)
    light = np.clip(light, 0, 85000)

    # CO2: higher at night, 350–1200 ppm
    co2 = 700 - 200 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 30, n_rows)
    co2 = np.clip(co2, 350, 1500)

    # Anomaly labels: ~5% anomalous windows
    labels = (rng.random(n_rows) < 0.05).astype(int)
    anom_idx = np.where(labels)[0]
    temp[anom_idx] += rng.choice([-10, 12], len(anom_idx))
    humidity[anom_idx] += rng.choice([-20, 25], len(anom_idx))
    temp = np.clip(temp, -5, 45)
    humidity = np.clip(humidity, 10, 99)

    timestamps = pd.date_range("2023-01-01", periods=n_rows, freq="15min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "soil_moisture_pct": soil,
        "light_lux": light,
        "co2_ppm": co2,
        "is_anomaly": labels,
    })


def split_timeseries(df: pd.DataFrame,
                     train_frac: float = 0.70,
                     val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split (no shuffle)."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def fit_and_save_scaler(train_df: pd.DataFrame, scaler_path: Path) -> StandardScaler:
    """Fit scaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(train_df[SENSOR_COLS])
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    return scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UCI greenhouse dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("./datasets/processed"))
    parser.add_argument("--n-synthetic", type=int, default=50000)
    parser.add_argument("--real-csv", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic greenhouse data...")
    df = generate_synthetic_greenhouse(args.n_synthetic)

    if args.real_csv and args.real_csv.exists():
        print(f"Loading real data: {args.real_csv}")
        real = pd.read_csv(args.real_csv)
        real = real.rename(columns={
            "AT": "temperature_c", "RH": "humidity_pct",
            "SM": "soil_moisture_pct", "LIGHT": "light_lux", "CO2": "co2_ppm"
        })
        if "is_anomaly" not in real.columns:
            real["is_anomaly"] = 0
        real = real[["timestamp"] + SENSOR_COLS + ["is_anomaly"]].dropna()
        df = pd.concat([df, real], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"✓ Combined: {len(df):,} rows")

    train, val, test = split_timeseries(df)
    scaler = fit_and_save_scaler(train, args.output_dir / "scaler.pkl")

    for split_df, name in [(train, "train"), (val, "val"), (test, "test")]:
        split_df.to_csv(args.output_dir / f"sensors_{name}.csv", index=False)
        print(f"✓ {name}: {len(split_df):,} rows → sensors_{name}.csv")

    print(f"\nTraining set stats:\n{train[SENSOR_COLS].describe().round(2)}")


if __name__ == "__main__":
    main()
