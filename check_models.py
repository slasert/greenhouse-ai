#!/usr/bin/env python3
"""
Validate trained models and show status.
"""

import sys
from pathlib import Path

def check_models():
    models_dir = Path("./models")
    
    models = {
        "LSTM": models_dir / "lstm" / "lstm_best.pth",
        "Autoencoder": models_dir / "autoencoder" / "autoencoder_best.pth",
        "PPO": models_dir / "ppo" / "best_model.zip",
        "CNN": models_dir / "cnn" / "cnn_best.pth",
        "Digital Twin": models_dir / "digital_twin" / "digital_twin_best.pth",
    }
    
    print("=" * 70)
    print("GREENHOUSE AI - TRAINED MODELS STATUS")
    print("=" * 70)
    
    trained = 0
    for name, path in models.items():
        exists = path.exists()
        status = "✓ TRAINED" if exists else "○ Pending"
        size = f"({path.stat().st_size / 1024:.0f} KB)" if exists else ""
        print(f"{name:15} {status:12} {size}")
        if exists:
            trained += 1
    
    print("=" * 70)
    print(f"Total: {trained}/{len(models)} models trained")
    
    # Dataset status
    print("\n" + "=" * 70)
    print("DATASET STATUS")
    print("=" * 70)
    
    data_files = [
        ("Train data", Path("datasets/processed/sensors_train.csv")),
        ("Val data", Path("datasets/processed/sensors_val.csv")),
        ("Test data", Path("datasets/processed/sensors_test.csv")),
        ("Scaler", Path("datasets/processed/scaler.pkl")),
    ]
    
    for name, path in data_files:
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = f"({path.stat().st_size / 1024:.0f} KB)" if exists else ""
        print(f"{status} {name:15} {size}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Check training logs:
   tail -50 logs/train_all_*.log

2. Integrate models into agent:
   from src.integration.model_registry import GreenhouseModelRegistry
   registry = GreenhouseModelRegistry()

3. Use trained models:
   forecast = registry.lstm.predict(sensor_window)
   is_anomaly = registry.autoencoder.is_anomaly(window)
   actions = registry.ppo.get_action(obs)

4. Run main application:
   streamlit run main.py
    """)

if __name__ == "__main__":
    check_models()
