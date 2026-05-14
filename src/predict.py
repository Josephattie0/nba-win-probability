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
# Calibration helper
# ---------------------------------------------------------------------------

def calibrate_model(csv_path: str, sample_n: int = 5000) -> dict:
    """
    Sample plays from the training CSV, run batched GRU predictions on
    zero-padded single-step sequences, bin into 10 buckets, and return
    calibration data + a human-readable summary.
    """
    global _model, _mean, _std
    if _model is None:
        _load()

    import pandas as pd
    from features import FEATURES

    df = pd.read_csv(csv_path)
    if "home_win" not in df.columns or not all(f in df.columns for f in FEATURES):
        raise ValueError("CSV is missing required columns — re-run collect.py first.")

    sample = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
    feats  = sample[FEATURES].values.astype(np.float32)
    labels = sample["home_win"].values.astype(np.float32)
    norm   = (feats - _mean) / _std

    # Zero-pad each row to SEQ_LEN → (n, SEQ_LEN, n_features)
    preds, batch_sz = [], 512
    _model.eval()
    for i in range(0, len(norm), batch_sz):
        chunk = norm[i : i + batch_sz]
        n     = len(chunk)
        pad   = np.zeros((n, SEQ_LEN - 1, norm.shape[1]), dtype=np.float32)
        seqs  = np.concatenate([pad, chunk[:, np.newaxis, :]], axis=1)
        with torch.no_grad():
            p = _model(torch.from_numpy(seqs)).numpy()
        preds.extend(p.tolist())

    preds = np.array(preds)

    buckets = []
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        mask = (preds >= lo) & (preds < hi)
        if not mask.any():
            continue
        buckets.append({
            "midpoint":  round((lo + hi) / 2, 2),
            "predicted": round(float(preds[mask].mean()), 3),
            "actual":    round(float(labels[mask].mean()), 3),
            "count":     int(mask.sum()),
        })

    devs      = [b["predicted"] - b["actual"] for b in buckets]
    max_dev   = max(abs(d) for d in devs) if devs else 0
    avg_dev   = sum(devs) / len(devs) if devs else 0
    close_b   = [b for b in buckets if 0.35 <= b["midpoint"] <= 0.65]
    close_dev = (sum(b["predicted"] - b["actual"] for b in close_b) / len(close_b)
                 if close_b else 0)

    if max_dev < 0.05:
        summary = "Model is well calibrated"
    elif max_dev < 0.10:
        summary = ("Slightly overconfident in close games" if close_dev > 0.03
                   else "Slightly underconfident in close games" if close_dev < -0.03
                   else "Slightly overconfident overall" if avg_dev > 0
                   else "Slightly underconfident overall")
    else:
        summary = ("Significantly overconfident in close games" if close_dev > 0.05
                   else "Significantly underconfident in close games" if close_dev < -0.05
                   else "Significant calibration error — consider retraining")

    return {"buckets": buckets, "summary": summary}


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
