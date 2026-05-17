"""
Unit tests for features.py

Covers: score parsing, clock parsing, seconds-left calculation,
foul bonus (regulation + OT thresholds + period reset), timeout
tracking (regulation + OT reset), rolling FG%, foul trouble
threshold crossing (technical fouls excluded), momentum window,
and an end-to-end build_features integration test.
"""

import pandas as pd
import pytest

from features import (
    FEATURES,
    _compute_bonus,
    _compute_fg_pct,
    _compute_foul_trouble,
    _compute_momentum,
    _compute_timeouts,
    _infer_teams,
    _parse_clock,
    _parse_scores,
    _seconds_left,
    build_features,
)

HOME, AWAY = "LAL", "GSW"
HOME_ID, AWAY_ID = "1610612747", "1610612744"


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _row(**kw):
    base = dict(
        gameid="22300001", actionnumber=1,
        clock="PT12M00.00S", period=1,
        teamtricode="", teamid="", personid="",
        location="", scorehome="", scoreaway="",
        actiontype="rebound", subtype="",
        isfieldgoal=0, shotresult="",
    )
    base.update(kw)
    return base


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Score parsing — forward-fill over blank (non-scoring) rows
# ---------------------------------------------------------------------------

def test_score_parsing_forward_fill():
    df = _df([
        _row(scorehome="0",  scoreaway="0"),
        _row(scorehome="",   scoreaway=""),     # non-scoring play
        _row(scorehome="3",  scoreaway="0"),
        _row(scorehome="",   scoreaway=""),     # non-scoring play
    ])
    _parse_scores(df)
    assert list(df["home_score"]) == [0, 0, 3, 3]
    assert list(df["score_diff"]) == [0, 0, 3, 3]


# ---------------------------------------------------------------------------
# 2. Clock parsing — "PTmmMss.SSS" → integer minutes / seconds
# ---------------------------------------------------------------------------

def test_clock_parsing():
    df = _df([_row(clock="PT06M30.00S")])
    _parse_clock(df)
    assert df["clock_minutes"].iloc[0] == 6
    assert df["clock_seconds"].iloc[0] == 30.0


def test_clock_parsing_zero():
    df = _df([_row(clock="PT00M00.00S")])
    _parse_clock(df)
    assert df["clock_minutes"].iloc[0] == 0
    assert df["clock_seconds"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# 3. Seconds remaining — regulation and OT
# ---------------------------------------------------------------------------

def test_seconds_left_start_of_game():
    row = _row(clock="PT12M00.00S", period=1)
    df = _df([row])
    _parse_clock(df)
    assert _seconds_left(df.iloc[0]) == pytest.approx(2880.0)   # 4 × 720


def test_seconds_left_mid_game():
    # Q3 with 6:00 left → 1 quarter remaining (720) + 360 = 1080
    row = _row(clock="PT06M00.00S", period=3)
    df = _df([row])
    _parse_clock(df)
    assert _seconds_left(df.iloc[0]) == pytest.approx(1080.0)


def test_seconds_left_overtime():
    # OT (period 5) with 2:30 left → only clock remaining counts
    row = _row(clock="PT02M30.00S", period=5)
    df = _df([row])
    _parse_clock(df)
    assert _seconds_left(df.iloc[0]) == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 4. Foul bonus — regulation threshold = 5
# ---------------------------------------------------------------------------

def test_bonus_triggers_at_threshold():
    """Home is in bonus once away commits 5 fouls in the quarter."""
    rows = []
    for i in range(6):
        rows.append(_row(
            actionnumber=i + 1,
            actiontype="Foul", subtype="Personal",
            teamtricode=AWAY, location="v",
        ))
    df = _df(rows)
    _compute_bonus(df, HOME, AWAY, HOME_ID, AWAY_ID)
    # before 5th away foul home_in_bonus should still be 0, after it should be 1
    assert df["home_in_bonus"].iloc[4] == 0   # 4 fouls — not yet
    assert df["home_in_bonus"].iloc[5] == 1   # 5th foul recorded — bonus


def test_bonus_resets_each_period():
    """Foul counts reset at the start of each quarter."""
    rows = [
        _row(actionnumber=1, actiontype="Foul", subtype="Personal",
             teamtricode=AWAY, period=1),
        _row(actionnumber=2, actiontype="Foul", subtype="Personal",
             teamtricode=AWAY, period=1),
        # new period — count should reset
        _row(actionnumber=3, actiontype="Foul", subtype="Personal",
             teamtricode=AWAY, period=2),
    ]
    df = _df(rows)
    _compute_bonus(df, HOME, AWAY, HOME_ID, AWAY_ID)
    # Only 2 fouls in Q1, never reached 5; Q2 starts fresh
    assert all(df["home_in_bonus"] == 0)


# ---------------------------------------------------------------------------
# 5. Foul bonus — OT threshold = 4
# ---------------------------------------------------------------------------

def test_bonus_ot_threshold_is_four():
    """In OT the bonus kicks in after 4 fouls, not 5."""
    rows = []
    for i in range(5):
        rows.append(_row(
            actionnumber=i + 1,
            actiontype="Foul", subtype="Personal",
            teamtricode=AWAY, period=5,   # first OT
        ))
    df = _df(rows)
    _compute_bonus(df, HOME, AWAY, HOME_ID, AWAY_ID)
    assert df["home_in_bonus"].iloc[3] == 0   # 3 fouls — not yet
    assert df["home_in_bonus"].iloc[4] == 1   # 4th foul — bonus in OT


# ---------------------------------------------------------------------------
# 6. Timeouts — decrement per call, reset to 3 at start of OT
# ---------------------------------------------------------------------------

def test_timeouts_decrement():
    rows = [
        _row(actionnumber=1, actiontype="Timeout", teamtricode=HOME, period=1),
        _row(actionnumber=2, actiontype="Timeout", teamtricode=HOME, period=1),
        _row(actionnumber=3, actiontype="rebound",  teamtricode="",   period=1),
    ]
    df = _df(rows)
    _compute_timeouts(df, HOME, AWAY, HOME_ID, AWAY_ID)
    # row 0: before first timeout → 7; row 1: after 1st → 6; row 2: after 2nd → 5
    assert df["home_timeouts"].iloc[0] == 7
    assert df["home_timeouts"].iloc[1] == 6
    assert df["home_timeouts"].iloc[2] == 5


def test_timeouts_reset_in_ot():
    """Entering OT resets each team's count to 3."""
    rows = [
        _row(actionnumber=1, actiontype="Timeout", teamtricode=HOME, period=4),
        _row(actionnumber=2, actiontype="rebound",  teamtricode="",   period=5),  # OT starts
    ]
    df = _df(rows)
    _compute_timeouts(df, HOME, AWAY, HOME_ID, AWAY_ID)
    # After the OT period boundary the count resets
    assert df["home_timeouts"].iloc[1] == 3


# ---------------------------------------------------------------------------
# 7. Rolling FG% — zero before first attempt, correct fraction after
# ---------------------------------------------------------------------------

def test_fg_pct_zero_before_first_attempt():
    df = _df([_row(actiontype="rebound")])
    _parse_scores(df)
    _compute_fg_pct(df, HOME, AWAY)
    assert df["home_fg_pct"].iloc[0] == 0.0


def test_fg_pct_rolling_accuracy():
    rows = [
        _row(actionnumber=1, actiontype="Made Shot",   isfieldgoal=1, teamtricode=HOME, location="h"),
        _row(actionnumber=2, actiontype="Missed Shot", isfieldgoal=1, teamtricode=HOME, location="h"),
        _row(actionnumber=3, actiontype="Made Shot",   isfieldgoal=1, teamtricode=HOME, location="h"),
        _row(actionnumber=4, actiontype="Made Shot",   isfieldgoal=1, teamtricode=HOME, location="h"),
    ]
    df = _df(rows)
    _parse_scores(df)
    _compute_fg_pct(df, HOME, AWAY)
    # Before any attempts: 0; after 1 made: 1.0; after 1 miss: 0.5; after 3/3: ...
    assert df["home_fg_pct"].iloc[0] == pytest.approx(0.0)
    assert df["home_fg_pct"].iloc[1] == pytest.approx(1.0)   # 1/1 before this row
    assert df["home_fg_pct"].iloc[2] == pytest.approx(0.5)   # 1/2
    assert df["home_fg_pct"].iloc[3] == pytest.approx(2/3)   # 2/3


# ---------------------------------------------------------------------------
# 8. Foul trouble — 4+ fouls triggers counter; technical fouls excluded
# ---------------------------------------------------------------------------

def test_foul_trouble_threshold_crossing():
    """A player crossing 4 personal fouls increments home_foul_trouble."""
    rows = []
    for i in range(5):
        rows.append(_row(
            actionnumber=i + 1,
            actiontype="Foul", subtype="Personal",
            teamtricode=HOME, personid="100",
        ))
    df = _df(rows)
    _compute_foul_trouble(df, HOME, AWAY)
    # Before 4th foul: 0; at and after 4th foul: 1
    assert df["home_foul_trouble"].iloc[3] == 0   # row index 3 = before 4th foul recorded
    assert df["home_foul_trouble"].iloc[4] == 1


def test_technical_foul_excluded_from_trouble():
    """Technical fouls must not count toward personal foul trouble."""
    rows = [
        _row(actionnumber=i + 1, actiontype="Foul", subtype="Technical",
             teamtricode=HOME, personid="200")
        for i in range(10)
    ]
    df = _df(rows)
    _compute_foul_trouble(df, HOME, AWAY)
    assert all(df["home_foul_trouble"] == 0)


# ---------------------------------------------------------------------------
# 9. Momentum — rolling 5-play window of +1 (home) / -1 (away) scoring events
# ---------------------------------------------------------------------------

def test_momentum_window():
    rows = [
        _row(actionnumber=1, scorehome="2",  scoreaway="0"),   # home +2
        _row(actionnumber=2, scorehome="2",  scoreaway="2"),   # away +2
        _row(actionnumber=3, scorehome="4",  scoreaway="2"),   # home +2
        _row(actionnumber=4, scorehome="4",  scoreaway="4"),   # away +2
        _row(actionnumber=5, scorehome="6",  scoreaway="4"),   # home +2  → window full: [1,-1,1,-1,1]=1
        _row(actionnumber=6, scorehome="8",  scoreaway="4"),   # home +2  → drops first +1 → [−1,1,−1,1,1]=1
        _row(actionnumber=7, scorehome="8",  scoreaway="6"),   # away +2  → [1,−1,1,1,−1]=1
    ]
    df = _df(rows)
    _parse_scores(df)
    _compute_momentum(df)
    # momentum is recorded BEFORE the row's scoring event is appended
    # iloc[0]: window=[]       → 0
    # iloc[1]: window=[1]      → 1
    # iloc[2]: window=[1,-1]   → 0
    # iloc[3]: window=[1,-1,1] → 1
    # iloc[4]: window=[1,-1,1,-1] → 0
    # iloc[5]: window=[1,-1,1,-1,1] → 1  (window now full at 5)
    assert df["momentum"].iloc[0] == 0
    assert df["momentum"].iloc[4] == 0
    assert df["momentum"].iloc[5] == 1


# ---------------------------------------------------------------------------
# 10. build_features integration — correct columns, no nulls, sensible values
# ---------------------------------------------------------------------------

def _minimal_game():
    """Smallest valid game: tip, a few plays, clear winner."""
    rows = [
        # tip-off (sets teams via location)
        _row(actionnumber=1,  actiontype="Jump Ball",  location="h", teamtricode=HOME, clock="PT12M00.00S", period=1, scorehome="", scoreaway=""),
        _row(actionnumber=2,  actiontype="Jump Ball",  location="v", teamtricode=AWAY, clock="PT12M00.00S", period=1, scorehome="", scoreaway=""),
        # home scores
        _row(actionnumber=3,  actiontype="Made Shot",  location="h", teamtricode=HOME, isfieldgoal=1, clock="PT11M30.00S", period=1, scorehome="2", scoreaway="0"),
        # away scores
        _row(actionnumber=4,  actiontype="Made Shot",  location="v", teamtricode=AWAY, isfieldgoal=1, clock="PT11M00.00S", period=1, scorehome="2", scoreaway="2"),
        # home scores again — ends Q1 up by 2
        _row(actionnumber=5,  actiontype="Made Shot",  location="h", teamtricode=HOME, isfieldgoal=1, clock="PT00M01.00S", period=1, scorehome="4", scoreaway="2"),
        # Q2 — home up at end of game
        _row(actionnumber=6,  actiontype="Made Shot",  location="h", teamtricode=HOME, isfieldgoal=1, clock="PT00M05.00S", period=4, scorehome="6", scoreaway="2"),
        _row(actionnumber=7,  actiontype="period",     location="",  teamtricode="",   clock="PT00M00.00S", period=4, scorehome="6", scoreaway="2"),
    ]
    # Return with original-style columns (build_features lowercases internally)
    df = pd.DataFrame(rows)
    df.columns = [c for c in df.columns]   # already lowercase — fine
    return df


def test_build_features_returns_correct_columns():
    df = build_features(_minimal_game())
    assert list(df.columns) == ["game_id", "actionnumber"] + FEATURES + ["home_win"]


def test_build_features_no_nulls():
    df = build_features(_minimal_game())
    assert df.isnull().sum().sum() == 0


def test_build_features_label_correct():
    """Home wins by 4 points — home_win label should be 1."""
    df = build_features(_minimal_game())
    assert df["home_win"].iloc[-1] == 1


def test_build_features_seconds_left_decreasing():
    """seconds_left must be non-increasing as the game progresses."""
    df = build_features(_minimal_game())
    sl = df["seconds_left"].tolist()
    assert all(sl[i] >= sl[i + 1] for i in range(len(sl) - 1))
