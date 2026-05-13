"""
Phase 2 — Inference helper.

Loads the trained model and scaler, then exposes a single function:

    predict(score_diff, seconds_left, home_possession, home_in_bonus, away_in_bonus)
    -> float  (home win probability, 0.0 – 1.0)

Used by server.py to serve predictions over the Flask/WebSocket API.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.json")


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
# Module-level singleton — loaded once on first import
# ---------------------------------------------------------------------------

_model: WinProbNet | None = None
_mean: np.ndarray | None = None
_std: np.ndarray | None = None


def _load() -> None:
    global _model, _mean, _std

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} — run train.py first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH} — run train.py first.")

    with open(SCALER_PATH) as f:
        scaler = json.load(f)
    _mean = np.array(scaler["mean"], dtype=np.float32)
    _std = np.array(scaler["std"], dtype=np.float32)

    _model = WinProbNet(n_features=len(_mean))
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    _model.eval()


def predict(
    score_diff: float,
    seconds_left: float,
    home_possession: int,
    home_in_bonus: int,
    away_in_bonus: int,
) -> float:
    """Return home team win probability as a float in [0, 1]."""
    global _model, _mean, _std
    if _model is None:
        _load()

    x = np.array([[score_diff, seconds_left, home_possession, home_in_bonus, away_in_bonus]], dtype=np.float32)
    x = (x - _mean) / _std

    with torch.no_grad():
        prob = _model(torch.from_numpy(x)).item()
    return round(prob, 4)


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        # score_diff, seconds_left, home_poss, home_bonus, away_bonus, description
        (0,   2880, 1, 0, 0, "Tip-off, tied"),
        (10,  60,   1, 0, 0, "Home up 10, 1 min left"),
        (-5,  120,  0, 1, 0, "Away up 5, 2 min left, home in bonus"),
        (0,   30,   1, 1, 1, "Tied, 30s left, both in bonus"),
        (20,  10,   1, 0, 0, "Home up 20, 10s left"),
    ]

    print(f"{'Description':<45} {'Home win %':>10}")
    print("-" * 57)
    for sd, sl, hp, hb, ab, desc in examples:
        p = predict(sd, sl, hp, hb, ab)
        print(f"{desc:<45} {p*100:>9.1f}%")
