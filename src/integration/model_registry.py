#!/usr/bin/env python3
"""
Unified model registry for loading all six trained models.
Usage:
    from src.integration.model_registry import GreenhouseModelRegistry
    registry = GreenhouseModelRegistry()
    forecast = registry.lstm.predict(window)
    is_anomaly = registry.autoencoder.is_anomaly(window)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import joblib


class LSTMPredictor:
    def __init__(self, model_path: Path, scaler_path: Path, device: str = "cuda") -> None:
        from train_all import LSTMForecaster
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LSTMForecaster().to(self.device)
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    def predict(self, window: np.ndarray) -> np.ndarray:
        """window: (24, 5) raw sensor values → returns (6, 5) forecast"""
        if self.scaler:
            window = self.scaler.transform(window)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]
        return self.scaler.inverse_transform(pred) if self.scaler else pred


class AutoencoderDetector:
    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        from train_all import SensorAutoencoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SensorAutoencoder().to(self.device)
        self.threshold = 0.05
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.threshold = ckpt.get("threshold", 0.05)
        self.model.eval()

    def is_anomaly(self, window: np.ndarray) -> bool:
        """window: (24, 5) → True if anomalous"""
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model.anomaly_score(x).item()
        return score > self.threshold


class PPOController:
    def __init__(self, model_path: Path) -> None:
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path)) if model_path.exists() else None
        except ImportError:
            self.model = None

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """obs: (11,) normalized observation → returns 5 binary actuators"""
        if self.model is None:
            return np.zeros(5, dtype=int)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array([(action >> i) & 1 for i in range(5)], dtype=int)


class GreenhouseModelRegistry:
    """Single point to load all trained models."""
    def __init__(self, model_dir: Path = Path("./models"),
                 data_dir: Path = Path("./datasets/processed"),
                 device: str = "cuda") -> None:
        scaler_path = data_dir / "scaler.pkl"
        self.lstm = LSTMPredictor(model_dir / "lstm" / "lstm_best.pth",
                                  scaler_path, device)
        self.autoencoder = AutoencoderDetector(model_dir / "autoencoder" / "autoencoder_best.pth",
                                               device)
        self.ppo = PPOController(model_dir / "ppo" / "best_model.zip")
