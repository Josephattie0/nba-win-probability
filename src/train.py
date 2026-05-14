"""
Phase 2 — GRU sequence model training.

Reads data/play_by_play.csv, groups plays into sequences of SEQ_LEN events
per game, trains a GRU win-probability model, and saves:
    data/model.pth     — trained weights
    data/scaler.json   — per-feature mean/std for inference normalization

Usage:
    python train.py
    python train.py --epochs 50 --lr 5e-4 --seq-len 20 --stride 10
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from features import FEATURES

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH   = os.path.join(DATA_DIR, "play_by_play.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pth")
SCALER_PATH= os.path.join(DATA_DIR, "scaler.json")

LABEL   = "home_win"
SEQ_LEN = 20
STRIDE  = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WinProbGRU(nn.Module):
    def __init__(self, n_features: int = len(FEATURES), hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, h = self.gru(x)          # h: (n_layers, batch, hidden)
        return self.head(h[-1]).squeeze(-1)   # (batch,)


# ---------------------------------------------------------------------------
# Lazy sequence dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Lazily produces (seq, label) pairs from a list of per-game arrays.
    Avoids materializing all sequences in RAM at once.
    """

    def __init__(self, game_X: list, game_y: list, seq_len: int, stride: int):
        self.game_X  = game_X
        self.game_y  = game_y
        self.seq_len = seq_len
        self.stride  = stride
        self._index  = self._build_index()

    def _build_index(self):
        idx = []
        for g, X in enumerate(self.game_X):
            n = len(X)
            if n < self.seq_len:
                idx.append((g, -1))   # will be zero-padded
            else:
                for s in range(0, n - self.seq_len + 1, self.stride):
                    idx.append((g, s))
        return idx

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        g, s = self._index[i]
        X_g, y_g = self.game_X[g], self.game_y[g]
        n = len(X_g)

        if s == -1:   # short game — pad front with zeros
            pad   = np.zeros((self.seq_len - n, X_g.shape[1]), dtype=np.float32)
            X_seq = np.vstack([pad, X_g])
            label = y_g[-1]
        else:
            X_seq = X_g[s : s + self.seq_len]
            label = y_g[s + self.seq_len - 1]

        return torch.from_numpy(X_seq), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_games(csv_path: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Returns (game_X_list, game_y_list, global_mean, global_std).
    Each element of game_X_list is a (n_events, n_features) float32 array,
    already normalised.  game_y_list elements are (n_events,) float32 arrays.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Compute scaler on all rows before splitting into games
    X_all = df[FEATURES].values.astype(np.float32)
    mean  = X_all.mean(axis=0)
    std   = X_all.std(axis=0)
    std   = np.where(std == 0, 1.0, std)

    order_col = "actionnumber" if "actionnumber" in df.columns else None

    game_X, game_y = [], []
    for _, gdf in df.groupby("game_id"):
        if order_col:
            gdf = gdf.sort_values(order_col)
        X = ((gdf[FEATURES].values.astype(np.float32) - mean) / std)
        y = gdf[LABEL].values.astype(np.float32)
        game_X.append(X)
        game_y.append(y)

    return game_X, game_y, mean, std


def save_scaler(mean: np.ndarray, std: np.ndarray) -> None:
    with open(SCALER_PATH, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES}, f)
    print(f"Scaler saved → {SCALER_PATH}")


# ---------------------------------------------------------------------------
# Train / val split (by game, not by row, to avoid leakage)
# ---------------------------------------------------------------------------

def split_games(game_X, game_y, val_frac=0.15, seed=42):
    rng    = np.random.default_rng(seed)
    n      = len(game_X)
    idx    = rng.permutation(n)
    split  = int(n * (1 - val_frac))
    tr, va = idx[:split], idx[split:]
    return ([game_X[i] for i in tr], [game_y[i] for i in tr],
            [game_X[i] for i in va], [game_y[i] for i in va])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(epochs=30, lr=5e-4, batch_size=256, seq_len=SEQ_LEN, stride=STRIDE, val_frac=0.15):
    print(f"Loading {CSV_PATH} …")
    game_X, game_y, mean, std = load_games(CSV_PATH)
    save_scaler(mean, std)

    total_events = sum(len(x) for x in game_X)
    home_wins    = sum(y[-1] for y in game_y)
    print(f"{len(game_X):,} games | {total_events:,} events | "
          f"{home_wins/len(game_y):.2%} home-win rate")

    tr_X, tr_y, va_X, va_y = split_games(game_X, game_y, val_frac)
    tr_ds = SequenceDataset(tr_X, tr_y, seq_len, stride)
    va_ds = SequenceDataset(va_X, va_y, seq_len, stride)
    print(f"Train seqs: {len(tr_ds):,} | Val seqs: {len(va_ds):,}")

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model     = WinProbGRU(n_features=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss    = float("inf")
    patience         = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(X_b)
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in va_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds   = model(X_b)
                va_loss += criterion(preds, y_b).item() * len(X_b)
                correct += ((preds >= 0.5) == y_b.bool()).sum().item()
        va_loss /= len(va_ds)
        va_acc   = correct / len(va_ds)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={tr_loss:.4f} | val={va_loss:.4f} | acc={va_acc:.2%}")

        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} | Model → {MODEL_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--seq-len",    type=int,   default=SEQ_LEN)
    p.add_argument("--stride",     type=int,   default=STRIDE)
    a = p.parse_args()
    train(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
          seq_len=a.seq_len, stride=a.stride)
