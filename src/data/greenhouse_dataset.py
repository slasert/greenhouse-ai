#!/usr/bin/env python3
"""
PyTorch Dataset classes for LSTM, Autoencoder, and CNN.
All load from preprocessed CSVs from uci_loader.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import joblib

SENSOR_COLS = ["temperature_c", "humidity_pct", "soil_moisture_pct",
               "light_lux", "co2_ppm"]


class SensorWindowDataset(Dataset):
    """Sliding window dataset for LSTM and Autoencoder."""

    def __init__(self, csv_path: Path, window: int = 24, horizon: int = 6,
                 scaler_path: Optional[Path] = None, task: str = "forecast") -> None:
        df = pd.read_csv(csv_path)
        self.task = task
        self.window = window
        self.horizon = horizon

        values = df[SENSOR_COLS].values.astype(np.float32)
        self.labels = df["is_anomaly"].values.astype(np.int64)

        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            values = scaler.transform(values).astype(np.float32)

        self.data = values

    def __len__(self) -> int:
        if self.task == "forecast":
            return len(self.data) - self.window - self.horizon + 1
        return len(self.data) - self.window + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.data[idx: idx + self.window])
        label = int(self.labels[idx + self.window - 1])

        if self.task == "forecast":
            y = torch.tensor(self.data[idx + self.window: idx + self.window + self.horizon])
            return {"x": x, "y": y, "label": label}
        else:
            return {"x": x, "label": label}


class PlantHealthDataset(Dataset):
    """Image dataset for CNN plant health classifier."""
    CLASSES = ["healthy", "nitrogen_deficiency", "overwatered", "underwatered", "disease"]

    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = Path(root) / split
        self.samples: List[Tuple[Path, int]] = []

        for cls_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, cls_idx))
                for img_path in cls_dir.glob("*.png"):
                    self.samples.append((img_path, cls_idx))

        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.RandomRotation(15)]
        self.train_tfm = T.Compose([T.Resize((224, 224))] + aug + [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_tfm = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.split = split

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            img = torch.randn(3, 224, 224)
            return {"image": img, "label": 0}
        path, label = self.samples[idx % len(self.samples)]
        img = Image.open(path).convert("RGB")
        tfm = self.train_tfm if self.split == "train" else self.val_tfm
        return {"image": tfm(img), "label": label}
