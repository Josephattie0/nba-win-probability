"""
Phase 2 — GRU inference helper.

Maintains a per-game rolling window of the last SEQ_LEN feature vectors
so each prediction uses a sequence, not a single snapshot.

Usage:
    from predict import predict
    prob = predict(game_id, score_diff, seconds_left, ...)
"""

import json
import os
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

from features import FEATURES

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH  = os.path.join(DATA_DIR, "model.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.json")
SEQ_LEN     = 20


# ---------------------------------------------------------------------------
# Model  (must match train.py exactly)
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
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Singleton model + scaler
# ---------------------------------------------------------------------------

_model: WinProbGRU | None = None
_mean:  np.ndarray | None = None
_std:   np.ndarray | None = None

# Per-game rolling window: game_id → deque of normalised feature vectors
_windows: dict = defaultdict(lambda: deque(maxlen=SEQ_LEN))


def _load() -> None:
    global _model, _mean, _std

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH} — run train.py first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"No scaler at {SCALER_PATH} — run train.py first.")

    with open(SCALER_PATH) as f:
        sc = json.load(f)
    _mean = np.array(sc["mean"], dtype=np.float32)
    _std  = np.array(sc["std"],  dtype=np.float32)

    n_feat = len(_mean)
    _model = WinProbGRU(n_features=n_feat)
    try:
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Model weights don't match architecture — re-run train.py.\n{e}"
        ) from e
    _model.eval()


def reset_game(game_id: str) -> None:
    """Clear the rolling window for a game (call when a new game starts)."""
    _windows.pop(game_id, None)


# ---------------------------------------------------------------------------
# Public predict function
# ---------------------------------------------------------------------------

def predict(
    game_id:          str,
    score_diff:       float,
    seconds_left:     float,
    home_possession:  int,
    home_in_bonus:    int,
    away_in_bonus:    int,
    home_timeouts:    int,
    away_timeouts:    int,
    home_fg_pct:      float,
    away_fg_pct:      float,
    home_foul_trouble: int,
    away_foul_trouble: int,
    momentum:         float,
) -> float:
    """
    Return home-team win probability in [0, 1].

    Internally appends the normalised feature vector to a per-game deque
    and runs the GRU on the last SEQ_LEN plays (zero-padded when < SEQ_LEN
    plays have been seen).
    """
    global _model, _mean, _std
    if _model is None:
        _load()

    raw = np.array([[
        score_diff, seconds_left, home_possession,
        home_in_bonus, away_in_bonus,
        home_timeouts, away_timeouts,
        home_fg_pct, away_fg_pct,
        home_foul_trouble, away_foul_trouble,
        momentum,
    ]], dtype=np.float32)

    norm = (raw - _mean) / _std   # (1, n_features)
    _windows[game_id].append(norm[0])

    # Build sequence, left-padding with zeros if the game just started
    window = list(_windows[game_id])
    if len(window) < SEQ_LEN:
        pad    = [np.zeros(len(FEATURES), dtype=np.float32)] * (SEQ_LEN - len(window))
        window = pad + window

    seq = torch.from_numpy(np.stack(window)).unsqueeze(0)  # (1, SEQ_LEN, n_features)
    with torch.no_grad():
        prob = _model(seq).item()
    return round(prob, 4)


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gid = "test_game"
    examples = [
        # score_diff, sec_left, poss, h_bonus, a_bonus, h_to, a_to,
        # h_fg, a_fg, h_ft, a_ft, momentum, desc
        (0,   2880, 1, 0, 0, 7, 7, 0.0, 0.0, 0, 0, 0,   "Tip-off, tied"),
        (10,   60,  1, 0, 0, 3, 4, 0.48, 0.43, 0, 0, 3, "Home up 10, 1 min left"),
        (-5,  120,  0, 1, 0, 4, 5, 0.41, 0.51, 2, 0, -2,"Away up 5, 2 min left"),
        (0,    30,  1, 1, 1, 1, 1, 0.46, 0.46, 1, 1, 0, "Tied, 30s, both bonus"),
        (20,   10,  1, 0, 0, 2, 3, 0.55, 0.44, 0, 0, 4, "Home up 20, 10s left"),
    ]

    print(f"{'Description':<40} {'Home win %':>10}")
    print("-" * 52)
    for row in examples:
        *feats, desc = row
        sd, sl, hp, hb, ab, hto, ato, hfg, afg, hft, aft, mom = feats
        p = predict(gid, sd, sl, hp, hb, ab, hto, ato, hfg, afg, hft, aft, mom)
        print(f"{desc:<40} {p*100:>9.1f}%")
