"""
Phase 2 — Model training.

Reads data/play_by_play.csv, trains a win-probability neural net, and saves:
    data/model.pth     — trained weights
    data/scaler.json   — feature mean/std for inference-time normalization

Usage:
    python train.py
    python train.py --epochs 50 --lr 0.001
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "play_by_play.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.json")

FEATURES = ["score_diff", "seconds_left", "home_possession", "home_in_bonus", "away_in_bonus"]
LABEL = "home_win"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WinProbNet(nn.Module):
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading & normalization
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL].values.astype(np.float32)
    return X, y


def normalize(X: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero on binary cols
    return (X - mean) / std, mean, std


def save_scaler(mean: np.ndarray, std: np.ndarray) -> None:
    with open(SCALER_PATH, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES}, f)
    print(f"Scaler saved to {SCALER_PATH}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(epochs: int = 30, lr: float = 5e-4, batch_size: int = 1024, val_frac: float = 0.15):
    print(f"Loading data from {CSV_PATH}")
    X, y = load_data(CSV_PATH)
    print(f"Dataset: {len(X):,} rows | label balance: {y.mean():.2%} home wins")

    # Shuffle and split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * (1 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    X_train, mean, std = normalize(X_train)
    X_val, _, _ = normalize(X_val, mean, std)
    save_scaler(mean, std)

    def to_loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model = WinProbNet(n_features=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(X_train)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_loss += criterion(preds, y_b).item() * len(X_b)
                correct += ((preds >= 0.5) == y_b.bool()).sum().item()
        val_loss /= len(X_val)
        val_acc = correct / len(X_val)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
