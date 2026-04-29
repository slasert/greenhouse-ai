#!/usr/bin/env python3
"""
train_all.py — Train all six AI models on UCI greenhouse sensor data.

Usage:
    python train_all.py --data-dir ./datasets/processed --model all --device cuda
    python train_all.py --model lstm
    python train_all.py --model ppo --ppo-timesteps 500000
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix encoding on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchdiffeq import odeint
from sklearn.metrics import roc_auc_score

from src.data.greenhouse_dataset import SensorWindowDataset, PlantHealthDataset

SENSOR_COLS = ["temperature_c", "humidity_pct", "soil_moisture_pct",
               "light_lux", "co2_ppm"]
N_SENSORS = 5


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: LSTM Sensor Forecaster
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    """Two-layer stacked LSTM for multi-step sensor forecasting."""
    def __init__(self, n_sensors: int = 5, hidden: int = 128,
                 n_layers: int = 2, window: int = 24, horizon: int = 6,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(n_sensors, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_sensors * horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, self.horizon, N_SENSORS)


def train_lstm(args, device, logger, wandb_run=None) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 1: LSTM Sensor Forecaster")
    logger.info("=" * 70)

    scaler_path = args.data_dir / "scaler.pkl"
    train_ds = SensorWindowDataset(args.data_dir / "sensors_train.csv",
                                   window=args.lstm_window, horizon=args.lstm_horizon,
                                   scaler_path=scaler_path, task="forecast")
    val_ds = SensorWindowDataset(args.data_dir / "sensors_val.csv",
                                 window=args.lstm_window, horizon=args.lstm_horizon,
                                 scaler_path=scaler_path, task="forecast")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = LSTMForecaster(window=args.lstm_window, horizon=args.lstm_horizon).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.HuberLoss(delta=1.0)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.lstm_epochs)
    scaler = GradScaler()
    best_val_loss, best_epoch = float("inf"), 0

    out_dir = args.output_dir / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.lstm_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            with autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        sched.step()

        # Validate every epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                with autocast():
                    val_loss += loss_fn(model(batch["x"].to(device)),
                                       batch["y"].to(device)).item()
        val_loss /= len(val_dl)
        avg_train = total_loss / len(train_dl)
        logger.info(f"[LSTM] Epoch {epoch}/{args.lstm_epochs} | "
                   f"Train: {avg_train:.5f} | Val: {val_loss:.5f}")
        if wandb_run:
            wandb_run.log({"lstm/train_loss": avg_train,
                          "lstm/val_loss": val_loss, "epoch": epoch})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                       "val_loss": val_loss},
                      str(out_dir / "lstm_best.pth"))
    
    # Save final model
    torch.save({"epoch": args.lstm_epochs, "model": model.state_dict(),
               "val_loss": best_val_loss},
              str(out_dir / "lstm_final.pth"))
    logger.info(f"[LSTM] Best epoch {best_epoch}, val_loss {best_val_loss:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: PPO Reinforcement Learning Agent
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo(args, device, logger, wandb_run=None) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 2: PPO Reinforcement Learning Agent")
    logger.info("=" * 70)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error("Install: pip install stable-baselines3 gymnasium")
        return

    from src.envs.greenhouse_env import GreenhouseEnv

    out_dir = args.output_dir / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(GreenhouseEnv, n_envs=4,
                       env_kwargs={"data_csv": args.data_dir / "sensors_train.csv"})
    eval_env = make_vec_env(GreenhouseEnv, n_envs=1,
                            env_kwargs={"data_csv": args.data_dir / "sensors_val.csv"})

    eval_cb = EvalCallback(eval_env, best_model_save_path=str(out_dir),
                          log_path=str(out_dir), eval_freq=5000,
                          deterministic=True, render=False)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device=str(device),
        tensorboard_log=str(out_dir / "tb_logs"),
    )

    logger.info(f"Training PPO for {args.ppo_timesteps:,} timesteps...")
    model.learn(total_timesteps=args.ppo_timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(out_dir / "ppo_agent"))
    logger.info(f"✓ PPO agent saved: {out_dir}/ppo_agent.zip")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CNN Plant Health Classifier
# ══════════════════════════════════════════════════════════════════════════════

class PlantHealthCNN(nn.Module):
    """EfficientNet-B0 fine-tuned for 5-class plant health classification."""
    CLASSES = ["healthy", "nitrogen_deficiency", "overwatered",
              "underwatered", "disease"]

    def __init__(self, n_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.features.parameters():
            p.requires_grad = True


def _eval_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / max(total, 1)


def train_cnn(args, device, logger, wandb_run=None) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 3: CNN Plant Health Classifier (EfficientNet-B0)")
    logger.info("=" * 70)

    image_dir = getattr(args, "image_dir", Path("./datasets/images"))
    train_ds = PlantHealthDataset(image_dir, split="train")
    val_ds = PlantHealthDataset(image_dir, split="val")

    if len(train_ds.samples) == 0:
        logger.warning("⚠ No plant images found. Skipping CNN training.")
        logger.warning("  Download PlantVillage: https://www.kaggle.com/emmarex/plantdisease")
        return

    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels, minlength=5)
    class_weights = torch.FloatTensor(
        len(labels) / (5 * class_counts + 1e-6)
    ).to(device)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                         num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = PlantHealthCNN(pretrained=True).to(device)
    model.freeze_backbone()

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    out_dir = args.output_dir / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler()
    best_acc = 0.0

    def run_epoch(phase: str, epochs: int, lr: float) -> None:
        nonlocal best_acc
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                              steps_per_epoch=len(train_dl),
                                              epochs=epochs)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in train_dl:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                with autocast():
                    logits = model(imgs)
                    loss = loss_fn(logits, labels)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)

            val_acc = _eval_cnn(model, val_dl, device)
            logger.info(f"[CNN {phase}] Epoch {epoch}/{epochs} | "
                       f"Loss: {total_loss/len(train_dl):.4f} | "
                       f"Train acc: {correct/total:.3f} | Val acc: {val_acc:.3f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "val_acc": val_acc},
                          str(out_dir / "cnn_best.pth"))

    run_epoch("head", epochs=5, lr=1e-3)
    model.unfreeze_backbone()
    run_epoch("finetune", epochs=15, lr=2e-4)
    logger.info(f"[CNN] Best val acc: {best_acc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: Bayesian Optimization (online)
# ══════════════════════════════════════════════════════════════════════════════

def demo_bayesian_optimizer(logger) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 4: Bayesian Optimization (online, no training)")
    logger.info("=" * 70)

    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        logger.error("Install: pip install scikit-optimize")
        return

    def greenhouse_objective(x: List[float]) -> float:
        schedule = [1 if xi > 0.5 else 0 for xi in x]
        irrigation_hours = sum(schedule)
        ideal_hours = 6
        irrigation_penalty = abs(irrigation_hours - ideal_hours) * 0.1
        timing_reward = sum(schedule[h] for h in range(4, 8)) * 0.15
        timing_reward += sum(schedule[h] for h in range(17, 21)) * 0.1
        consecutive = sum(
            1 for i in range(len(schedule)-1)
            if schedule[i] == 1 and schedule[i+1] == 1
        ) * 0.05
        score = 0.8 - irrigation_penalty + timing_reward - consecutive
        return -score

    space = [Real(0.0, 1.0, name=f"h{i}") for i in range(24)]
    result = gp_minimize(greenhouse_objective, space,
                        n_calls=60, n_initial_points=10,
                        acq_func="EI", random_state=42, verbose=False)
    best_schedule = [1 if xi > 0.5 else 0 for xi in result.x]
    logger.info(f"Bayesian best schedule: {best_schedule}")
    logger.info(f"✓ Best score: {-result.fun:.4f}")
    logger.info("  Integrate: replace GeneticAlgorithm.optimize() with gp_minimize()")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: Autoencoder Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════

class SensorAutoencoder(nn.Module):
    """LSTM-based Autoencoder for anomaly detection."""
    def __init__(self, n_sensors: int = 5, window: int = 24,
                 hidden: int = 64, latent: int = 16) -> None:
        super().__init__()
        self.window = window
        self.n_sensors = n_sensors
        self.encoder_lstm = nn.LSTM(n_sensors, hidden, 2,
                                   batch_first=True, dropout=0.1)
        self.enc_fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.decoder_lstm = nn.LSTM(hidden, hidden, 2,
                                   batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden, n_sensors)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder_lstm(x)
        return self.enc_fc(h[-1])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h0 = self.dec_fc(z).unsqueeze(0).repeat(2, 1, 1)
        dec_input = h0[-1].unsqueeze(1).repeat(1, self.window, 1)
        out, _ = self.decoder_lstm(dec_input, (h0, torch.zeros_like(h0)))
        return self.out(out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])


def train_autoencoder(args, device, logger, wandb_run=None) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 5: Autoencoder Anomaly Detector")
    logger.info("=" * 70)

    scaler_path = args.data_dir / "scaler.pkl"
    train_ds = SensorWindowDataset(args.data_dir / "sensors_train.csv",
                                  window=args.ae_window, scaler_path=scaler_path,
                                  task="reconstruct")
    val_ds = SensorWindowDataset(args.data_dir / "sensors_val.csv",
                                window=args.ae_window, scaler_path=scaler_path,
                                task="reconstruct")

    normal_idx = [i for i in range(len(train_ds))
                  if train_ds.labels[i + train_ds.window - 1] == 0]
    train_normal = Subset(train_ds, normal_idx)
    logger.info(f"Training on {len(train_normal):,} normal windows "
               f"({len(normal_idx)/len(train_ds)*100:.1f}% of data)")

    train_dl = DataLoader(train_normal, batch_size=128, shuffle=True,
                         num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    model = SensorAutoencoder(window=args.ae_window).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    scaler = GradScaler()
    out_dir = args.output_dir / "autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.ae_epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            x = batch["x"].to(device)
            with autocast():
                recon, _ = model(x)
                loss = F.mse_loss(recon, x)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        # Validate every epoch
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                scores = model.anomaly_score(batch["x"].to(device))
                val_scores.extend(scores.cpu().numpy())
                val_labels.extend(batch["label"].numpy())

        try:
            auroc = roc_auc_score(val_labels, val_scores)
        except Exception:
            auroc = 0.0

        avg_loss = total / len(train_dl)
        sched.step(avg_loss)
        logger.info(f"[AE] Epoch {epoch}/{args.ae_epochs} | "
                   f"Loss: {avg_loss:.6f} | Val AUROC: {auroc:.4f}")
        if wandb_run:
            wandb_run.log({"ae/train_loss": avg_loss,
                          "ae/val_auroc": auroc, "epoch": epoch})
        if avg_loss < best_loss:
            best_loss = avg_loss
            threshold = float(np.percentile(val_scores, 95))
            torch.save({"model": model.state_dict(),
                       "threshold": threshold,
                       "auroc": auroc},
                      str(out_dir / "autoencoder_best.pth"))
    
    # Save final model
    torch.save({"model": model.state_dict(),
               "threshold": threshold,
               "auroc": auroc},
              str(out_dir / "autoencoder_final.pth"))
    logger.info(f"[AE] Best loss {best_loss:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6: Digital Twin (Neural ODE)
# ══════════════════════════════════════════════════════════════════════════════

class GreenhouseODE(nn.Module):
    """Neural ODE for greenhouse thermodynamics."""
    def __init__(self, n_states: int = 5, n_actuators: int = 5,
                 hidden: int = 64) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.gamma = nn.Parameter(torch.tensor(0.05))
        self.delta = nn.Parameter(torch.tensor(0.08))
        self.zeta = nn.Parameter(torch.tensor(0.06))

        self.net = nn.Sequential(
            nn.Linear(n_states + n_actuators + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_states),
        )
        self.register_buffer("actuators", torch.zeros(n_actuators))
        self.t_external = 20.0

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        temp, hum, soil, light, co2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
        u = self.actuators

        dT = self.alpha * (self.t_external - temp) + self.beta * u[1] - self.gamma * u[2]
        dH = -0.02 * hum + self.delta * u[0] - self.zeta * u[3] + 0.01
        dS = -0.05 * soil + self.delta * u[0] * 2.0
        dL = -0.1 * (light - u[4] * 40000)
        dC = -0.02 * (co2 - 400) - self.zeta * u[3] * 50

        physics = torch.stack([dT, dH, dS, dL, dC], dim=-1)

        t_feat = t.expand(s.shape[:-1] + (1,)) if s.dim() > 1 else t.unsqueeze(0).unsqueeze(0)
        inp = torch.cat([s, self.actuators.expand_as(s[..., :5]),
                        t_feat.squeeze().unsqueeze(-1).expand_as(s[..., :1])], dim=-1)
        residual = self.net(inp) * 0.1

        return physics + residual


class DigitalTwin(nn.Module):
    """Full Digital Twin simulator."""
    def __init__(self) -> None:
        super().__init__()
        self.ode = GreenhouseODE()

    def simulate(self, s0: torch.Tensor, actuators: torch.Tensor,
                 n_steps: int = 24, dt: float = 0.25) -> torch.Tensor:
        self.ode.actuators = actuators
        t_span = torch.linspace(0, n_steps * dt, n_steps + 1).to(s0.device)
        traj = odeint(self.ode, s0.unsqueeze(0), t_span, method="rk4")
        return traj.squeeze(1)[1:]

    def forward(self, s0: torch.Tensor, actuators: torch.Tensor,
                n_steps: int = 24) -> torch.Tensor:
        return self.simulate(s0, actuators, n_steps)


def train_digital_twin(args, device, logger, wandb_run=None) -> None:
    logger.info("=" * 70)
    logger.info("MODEL 6: Digital Twin (Neural ODE)")
    logger.info("=" * 70)

    scaler_path = args.data_dir / "scaler.pkl"
    ds = SensorWindowDataset(args.data_dir / "sensors_train.csv",
                            window=args.twin_window, horizon=args.twin_horizon,
                            scaler_path=scaler_path, task="forecast")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = DigitalTwin().to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.twin_epochs)
    out_dir = args.output_dir / "digital_twin"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.twin_epochs + 1):
        model.train()
        total = 0.0
        for batch in dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            s0 = x[:, -1, :]
            u = torch.randint(0, 2, (s0.size(0), 5)).float().to(device)

            loss_batch = torch.tensor(0.0, device=device)
            for b in range(s0.size(0)):
                traj = model(s0[b], u[b], n_steps=args.twin_horizon)
                loss_batch = loss_batch + F.mse_loss(traj, y[b])
            loss = loss_batch / s0.size(0)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        sched.step()
        avg = total / len(dl)
        logger.info(f"[Twin] Epoch {epoch}/{args.twin_epochs} | Loss: {avg:.6f}")
        if wandb_run:
            wandb_run.log({"twin/train_loss": avg, "epoch": epoch})
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model": model.state_dict()},
                      str(out_dir / "digital_twin_best.pth"))

    torch.save(model.state_dict(), str(out_dir / "digital_twin_final.pth"))
    logger.info(f"[Twin] Best loss {best_loss:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all 6 greenhouse AI models")
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets/processed"))
    parser.add_argument("--image-dir", type=Path, default=Path("./datasets/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("./models"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="all",
                       choices=["all", "lstm", "ppo", "cnn", "bayesian",
                               "autoencoder", "digital_twin"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument("--lstm-window", type=int, default=24)
    parser.add_argument("--lstm-horizon", type=int, default=6)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-window", type=int, default=24)
    parser.add_argument("--ppo-timesteps", type=int, default=500_000)
    parser.add_argument("--twin-epochs", type=int, default=80)
    parser.add_argument("--twin-window", type=int, default=24)
    parser.add_argument("--twin-horizon", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs") / f"train_all_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(exist_ok=True)
    
    # Fix logging encoding on Windows
    class UTF8StreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                msg = msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), UTF8StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                   f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                  name="greenhouse_6models", config=vars(args))
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    run = args.model
    if run in ("all", "lstm"):
        train_lstm(args, device, logger, wandb_run)
    if run in ("all", "autoencoder"):
        train_autoencoder(args, device, logger, wandb_run)
    if run in ("all", "cnn"):
        train_cnn(args, device, logger, wandb_run)
    if run in ("all", "bayesian"):
        demo_bayesian_optimizer(logger)
    if run in ("all", "digital_twin"):
        train_digital_twin(args, device, logger, wandb_run)
    if run in ("all", "ppo"):
        train_ppo(args, device, logger, wandb_run)

    if wandb_run:
        wandb_run.finish()
    logger.info("✓ All training complete.")


if __name__ == "__main__":
    main()
