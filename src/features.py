"""
Phase 1 — Feature engineering.

Takes a raw PlayByPlayV3 DataFrame for one game and returns a DataFrame where
each row is one play-by-play event with these columns:

    game_id          str   — NBA game ID
    score_diff       int   — home score minus away score at this moment
    seconds_left     float — seconds remaining in the game (0 at buzzer)
    home_possession  int   — 1 if home team has the ball, 0 if away
    home_in_bonus    int   — 1 if home team is in the foul bonus
    away_in_bonus    int   — 1 if away team is in the foul bonus
    home_win         int   — 1 if the home team won, 0 if they lost (label)
"""

import pandas as pd

REGULATION_QUARTERS = 4
QUARTER_SECONDS = 12 * 60   # 720 s
OT_SECONDS = 5 * 60         # 300 s
BONUS_THRESHOLD = 5         # team fouls in regulation; 4 in OT


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]  # normalize to lowercase

    _parse_scores(df)
    _parse_clock(df)
    df["seconds_left"] = df.apply(_seconds_left, axis=1)
    df["home_possession"] = df["location"].apply(
        lambda loc: 1 if str(loc).lower() == "h" else 0
    )

    home_team, away_team = _infer_teams(df)
    _compute_bonus(df, home_team, away_team)

    label = _determine_winner(df)
    if label is None:
        raise ValueError("Could not determine game winner — skipping.")

    df["home_win"] = label
    df["game_id"] = df["gameid"].astype(str)

    out = df[[
        "game_id",
        "score_diff",
        "seconds_left",
        "home_possession",
        "home_in_bonus",
        "away_in_bonus",
        "home_win",
    ]].dropna()

    # Drop plays with no remaining time AND tied score (unresolvable)
    out = out[~((out["seconds_left"] == 0) & (out["score_diff"] == 0))]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _parse_scores(df: pd.DataFrame) -> None:
    # PlayByPlayV3 has separate scoreHome / scoreAway columns; forward-fill
    # because scoring events leave them blank on non-scoring plays.
    df["home_score"] = pd.to_numeric(df.get("scorehome", pd.Series()), errors="coerce").ffill().fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df.get("scoreaway", pd.Series()), errors="coerce").ffill().fillna(0).astype(int)
    df["score_diff"] = df["home_score"] - df["away_score"]


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def _parse_clock(df: pd.DataFrame) -> None:
    # Format: "PT06M30.00S"
    parts = df.get("clock", pd.Series(dtype=str)).str.extract(r"PT(\d+)M([\d.]+)S")
    df["clock_minutes"] = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    df["clock_seconds"] = pd.to_numeric(parts[1], errors="coerce").fillna(0)


def _seconds_left(row) -> float:
    period = int(row.get("period", 1) or 1)
    clock_remaining = row.get("clock_minutes", 0) * 60 + row.get("clock_seconds", 0)

    if period <= REGULATION_QUARTERS:
        return (REGULATION_QUARTERS - period) * QUARTER_SECONDS + clock_remaining
    else:
        # OT: we only know time left in the current OT period
        return float(clock_remaining)


# ---------------------------------------------------------------------------
# Team identification
# ---------------------------------------------------------------------------

def _infer_teams(df: pd.DataFrame) -> tuple[str, str]:
    """Return (home_tricode, away_tricode) by scanning location + teamtricode."""
    home_team = ""
    away_team = ""
    for _, row in df.iterrows():
        loc = str(row.get("location", "")).lower()
        tri = str(row.get("teamtricode", "") or "").upper()
        if not tri:
            continue
        if loc == "h" and not home_team:
            home_team = tri
        elif loc == "v" and not away_team:
            away_team = tri
        if home_team and away_team:
            break
    return home_team, away_team


# ---------------------------------------------------------------------------
# Foul bonus
# ---------------------------------------------------------------------------

def _compute_bonus(df: pd.DataFrame, home_team: str, away_team: str) -> None:
    """
    Add home_in_bonus and away_in_bonus (1/0).

    Home is in the bonus when the AWAY team has >= threshold fouls that quarter,
    and vice-versa (bonus = opponent gets free throws on any foul).
    Reset foul counts each period.
    """
    home_fouls = 0
    away_fouls = 0
    current_period = None

    home_in_bonus_list = []
    away_in_bonus_list = []

    for _, row in df.iterrows():
        period = row.get("period", 1) or 1
        if period != current_period:
            home_fouls = 0
            away_fouls = 0
            current_period = period

        is_ot = int(period) > REGULATION_QUARTERS
        threshold = 4 if is_ot else BONUS_THRESHOLD

        # Bonus state before this event
        home_in_bonus_list.append(1 if away_fouls >= threshold else 0)
        away_in_bonus_list.append(1 if home_fouls >= threshold else 0)

        action = str(row.get("actiontype", "") or "").lower()
        if "foul" not in action:
            continue

        foul_team = str(row.get("teamtricode", "") or "").upper()
        if foul_team == home_team:
            home_fouls += 1
        elif foul_team == away_team:
            away_fouls += 1

    df["home_in_bonus"] = home_in_bonus_list
    df["away_in_bonus"] = away_in_bonus_list


# ---------------------------------------------------------------------------
# Game winner
# ---------------------------------------------------------------------------

def _determine_winner(df: pd.DataFrame) -> int | None:
    """1 = home won, 0 = away won, None = could not determine."""
    # Walk backwards to find the last row with an actual score
    for _, row in df.iloc[::-1].iterrows():
        diff = row.get("score_diff")
        if pd.notna(diff):
            diff = int(diff)
            if diff > 0:
                return 1
            if diff < 0:
                return 0
    return None
