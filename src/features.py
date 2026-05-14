"""
Phase 1 — Feature engineering.

Produces one row per PBP event with these columns:

    game_id, actionnumber          — identity / ordering
    score_diff                     — home minus away points
    seconds_left                   — seconds remaining in game
    home_possession                — 1 = home has ball
    home_in_bonus / away_in_bonus  — 1 = team is in the foul bonus
    home_timeouts / away_timeouts  — timeouts remaining (7 reg, 3 OT)
    home_fg_pct / away_fg_pct      — rolling in-game FG% (0 if no attempts)
    home_foul_trouble              — # home players with 4+ personal fouls
    away_foul_trouble              — # away players with 4+ personal fouls
    momentum                       — last-5 scoring possessions (+1 home, -1 away)
    home_win                       — label (1 = home won)
"""

import pandas as pd

REGULATION_QUARTERS = 4
QUARTER_SECONDS = 12 * 60
OT_SECONDS      = 5  * 60
BONUS_THRESHOLD = 5          # team fouls → bonus in regulation (4 in OT)
FOUL_TROUBLE_THRESHOLD = 4   # personal fouls → "foul trouble"
MOMENTUM_WINDOW = 5          # last N scoring possessions

FEATURES = [
    "score_diff",
    "seconds_left",
    "home_possession",
    "home_in_bonus",
    "away_in_bonus",
    "home_timeouts",
    "away_timeouts",
    "home_fg_pct",
    "away_fg_pct",
    "home_foul_trouble",
    "away_foul_trouble",
    "momentum",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]

    home_team, away_team, home_id, away_id = _infer_teams(df)

    _parse_scores(df)
    _parse_clock(df)
    df["seconds_left"]    = df.apply(_seconds_left, axis=1)
    df["home_possession"] = df["location"].apply(lambda l: 1 if str(l).lower() == "h" else 0)

    _compute_bonus(df, home_team, away_team)
    _compute_timeouts(df, home_team, away_team, home_id, away_id)
    _compute_fg_pct(df, home_team, away_team)
    _compute_foul_trouble(df, home_team, away_team)
    _compute_momentum(df)

    label = _determine_winner(df)
    if label is None:
        raise ValueError("Could not determine game winner — skipping.")

    df["home_win"] = label
    df["game_id"]  = df["gameid"].astype(str)

    keep = ["game_id", "actionnumber"] + FEATURES + ["home_win"]
    out  = df[keep].dropna()
    out  = out[~((out["seconds_left"] == 0) & (out["score_diff"] == 0))]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score / clock helpers
# ---------------------------------------------------------------------------

def _parse_scores(df: pd.DataFrame) -> None:
    df["home_score"] = pd.to_numeric(df.get("scorehome", pd.Series()), errors="coerce").ffill().fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df.get("scoreaway", pd.Series()), errors="coerce").ffill().fillna(0).astype(int)
    df["score_diff"] = df["home_score"] - df["away_score"]


def _parse_clock(df: pd.DataFrame) -> None:
    parts = df.get("clock", pd.Series(dtype=str)).str.extract(r"PT(\d+)M([\d.]+)S")
    df["clock_minutes"] = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    df["clock_seconds"] = pd.to_numeric(parts[1], errors="coerce").fillna(0)


def _seconds_left(row) -> float:
    period          = int(row.get("period", 1) or 1)
    clock_remaining = row.get("clock_minutes", 0) * 60 + row.get("clock_seconds", 0)
    if period <= REGULATION_QUARTERS:
        return (REGULATION_QUARTERS - period) * QUARTER_SECONDS + clock_remaining
    return float(clock_remaining)


# ---------------------------------------------------------------------------
# Team identification
# ---------------------------------------------------------------------------

def _infer_teams(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Return (home_tri, away_tri, home_id, away_id)."""
    home_team, away_team, home_id, away_id = "", "", "", ""
    for _, row in df.iterrows():
        loc = str(row.get("location", "")).lower()
        tri = str(row.get("teamtricode", "") or "").upper()
        tid = str(row.get("teamid", "") or "")
        if not tri:
            continue
        if loc == "h" and not home_team:
            home_team, home_id = tri, tid
        elif loc == "v" and not away_team:
            away_team, away_id = tri, tid
        if home_team and away_team:
            break
    return home_team, away_team, home_id, away_id


def _team_of_row(row, home_team, away_team, home_id, away_id) -> str:
    """Return 'H', 'A', or '' for which team is associated with this event."""
    tri = str(row.get("teamtricode", "") or "").upper()
    if tri == home_team:
        return "H"
    if tri == away_team:
        return "A"
    # Fallback for events where teamtricode is blank (e.g. timeouts):
    # personid holds the team ID for those events.
    pid = str(row.get("personid", "") or "")
    if home_id and pid == home_id:
        return "H"
    if away_id and pid == away_id:
        return "A"
    return ""


# ---------------------------------------------------------------------------
# Foul bonus
# ---------------------------------------------------------------------------

def _compute_bonus(df: pd.DataFrame, home_team: str, away_team: str) -> None:
    home_fouls, away_fouls = 0, 0
    current_period = None
    home_in_bonus_list, away_in_bonus_list = [], []

    for _, row in df.iterrows():
        period = row.get("period", 1) or 1
        if period != current_period:
            home_fouls = away_fouls = 0
            current_period = period

        threshold = 4 if int(period) > REGULATION_QUARTERS else BONUS_THRESHOLD
        home_in_bonus_list.append(1 if away_fouls >= threshold else 0)
        away_in_bonus_list.append(1 if home_fouls >= threshold else 0)

        if "foul" not in str(row.get("actiontype", "") or "").lower():
            continue
        team = str(row.get("teamtricode", "") or "").upper()
        if team == home_team:
            home_fouls += 1
        elif team == away_team:
            away_fouls += 1

    df["home_in_bonus"] = home_in_bonus_list
    df["away_in_bonus"] = away_in_bonus_list


# ---------------------------------------------------------------------------
# Timeouts remaining
# ---------------------------------------------------------------------------

def _compute_timeouts(df, home_team, away_team, home_id, away_id) -> None:
    """
    Start each team at 7 for regulation. Reset to 3 when an OT period begins.
    Timeouts events have blank teamtricode but personid == team ID.
    """
    home_to = away_to = 7
    current_period = None
    home_list, away_list = [], []

    for _, row in df.iterrows():
        period = int(row.get("period", 1) or 1)

        if period != current_period:
            if current_period is not None and period > REGULATION_QUARTERS:
                home_to = away_to = 3   # each OT period resets to 3
            current_period = period

        home_list.append(home_to)
        away_list.append(away_to)

        if str(row.get("actiontype", "") or "").lower() != "timeout":
            continue

        side = _team_of_row(row, home_team, away_team, home_id, away_id)
        if side == "H":
            home_to = max(0, home_to - 1)
        elif side == "A":
            away_to = max(0, away_to - 1)

    df["home_timeouts"] = home_list
    df["away_timeouts"] = away_list


# ---------------------------------------------------------------------------
# Rolling FG%
# ---------------------------------------------------------------------------

def _compute_fg_pct(df, home_team, away_team) -> None:
    """Track cumulative field-goal percentage per team (free throws excluded)."""
    home_fgm = home_fga = away_fgm = away_fga = 0
    home_list, away_list = [], []

    for _, row in df.iterrows():
        home_list.append(home_fgm / home_fga if home_fga else 0.0)
        away_list.append(away_fgm / away_fga if away_fga else 0.0)

        if not int(row.get("isfieldgoal", 0) or 0):
            continue

        action = str(row.get("actiontype", "") or "").lower()
        team   = str(row.get("teamtricode", "") or "").upper()
        made   = action == "made shot"

        if team == home_team:
            home_fga += 1
            if made:
                home_fgm += 1
        elif team == away_team:
            away_fga += 1
            if made:
                away_fgm += 1

    df["home_fg_pct"] = home_list
    df["away_fg_pct"] = away_list


# ---------------------------------------------------------------------------
# Foul trouble (players with 4+ personal fouls)
# ---------------------------------------------------------------------------

def _compute_foul_trouble(df, home_team, away_team) -> None:
    """
    Count how many players on each team have >= FOUL_TROUBLE_THRESHOLD personal fouls.
    Maintains a running total that increments only when a player crosses the threshold.
    Technical fouls are excluded.
    """
    player_fouls: dict[str, int] = {}
    player_team:  dict[str, str] = {}
    home_trouble = away_trouble = 0
    home_list, away_list = [], []

    for _, row in df.iterrows():
        home_list.append(home_trouble)
        away_list.append(away_trouble)

        if "foul" not in str(row.get("actiontype", "") or "").lower():
            continue
        if "technical" in str(row.get("subtype", "") or "").lower():
            continue

        team      = str(row.get("teamtricode", "") or "").upper()
        person_id = str(row.get("personid", "") or "")
        if not person_id or not team:
            continue
        if team not in (home_team, away_team):
            continue

        player_team.setdefault(person_id, team)
        old = player_fouls.get(person_id, 0)
        new = old + 1
        player_fouls[person_id] = new

        if old < FOUL_TROUBLE_THRESHOLD <= new:
            if team == home_team:
                home_trouble += 1
            else:
                away_trouble += 1

    df["home_foul_trouble"] = home_list
    df["away_foul_trouble"] = away_list


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def _compute_momentum(df: pd.DataFrame) -> None:
    """
    Rolling sum over the last MOMENTUM_WINDOW scoring possessions.
    +1 each time home scores, -1 each time away scores.
    Score is read from the running home_score / away_score columns.
    """
    from collections import deque
    window: deque = deque(maxlen=MOMENTUM_WINDOW)
    prev_home = prev_away = 0
    momentum_list = []

    for _, row in df.iterrows():
        momentum_list.append(sum(window))

        h = int(row.get("home_score", 0) or 0)
        a = int(row.get("away_score", 0) or 0)
        if h > prev_home:
            window.append(1)
        elif a > prev_away:
            window.append(-1)
        prev_home, prev_away = h, a

    df["momentum"] = momentum_list


# ---------------------------------------------------------------------------
# Game winner
# ---------------------------------------------------------------------------

def _determine_winner(df: pd.DataFrame) -> int | None:
    for _, row in df.iloc[::-1].iterrows():
        diff = row.get("score_diff")
        if pd.notna(diff):
            diff = int(diff)
            if diff > 0:
                return 1
            if diff < 0:
                return 0
    return None
