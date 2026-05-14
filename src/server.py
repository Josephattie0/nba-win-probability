"""
Phase 3 — Flask + WebSocket server.

REST:
    GET  /games         — today's games from the live NBA scoreboard
    POST /predict       — one-shot prediction from a JSON game-state body

WebSocket (flask-socketio):
    client emits  subscribe    {game_id}  — start receiving live updates
    client emits  unsubscribe  {game_id}  — stop receiving updates
    server emits  game_update  {...}       — pushed every POLL_INTERVAL seconds

Usage:
    python server.py
"""

import os
import re
import threading
import time

import requests
from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO, emit

from predict import predict as model_predict

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

POLL_INTERVAL = 5  # seconds between live pushes

NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
    "Origin": "https://www.nba.com",
}

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"

QUARTER_SECONDS = 720   # 12 min regulation quarter
OT_SECONDS = 300        # 5 min OT
BONUS_THRESHOLD = 5     # team fouls before bonus (4 in OT)

# game_id -> set of connected socket session IDs
_subscriptions: dict[str, set] = {}
_sub_lock = threading.Lock()

# per-game foul tracking: game_id -> {"period", "home_fouls", "away_fouls", "home_team", "away_team", "seen_actions"}
_game_state: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# NBA data helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=NBA_HEADERS, timeout=5)
        if r.status_code == 200 and r.text.strip():
            return r.json()
    except Exception:
        pass
    return None


def _clock_to_seconds(game_clock: str, period: int) -> float:
    m = re.match(r"PT(\d+)M([\d.]+)S", game_clock or "")
    if not m:
        return 0.0
    remaining = float(m.group(1)) * 60 + float(m.group(2))
    if period <= 4:
        return (4 - period) * QUARTER_SECONDS + remaining
    return remaining  # OT: just time left in current period


def _fetch_scoreboard() -> list[dict]:
    data = _get(SCOREBOARD_URL)
    if not data:
        return []
    return data.get("scoreboard", {}).get("games", [])


def _fetch_pbp(game_id: str) -> list[dict]:
    data = _get(PBP_URL.format(game_id))
    if not data:
        return []
    return data.get("game", {}).get("actions", [])


# ---------------------------------------------------------------------------
# Feature extraction from live data
# ---------------------------------------------------------------------------

def _update_game_state(state: dict, actions: list[dict]) -> None:
    """
    Incrementally update all per-game tracking state from new live PBP actions.

    Live PBP field semantics:
      possession  — numeric team ID; for shots = shooting team, for fouls = fouled team
      actionType  — '2pt'|'3pt'|'freethrow'|'foul'|'timeout'|'period'|...
      shotResult  — 'Made' | 'Missed'
      subType     — foul sub-category
      personIdsFilter — list of personIds involved (index 0 = primary actor)
    """
    seen        = state.get("seen_actions", 0)
    new_actions = actions[seen:]
    state["seen_actions"] = len(actions)

    home_id = state.get("home_id", 0)
    away_id = state.get("away_id", 0)

    for action in new_actions:
        period = action.get("period", state.get("period", 1))

        # Period boundary — reset per-period counters
        if period != state.get("period"):
            state["period"]     = period
            state["home_fouls"] = 0
            state["away_fouls"] = 0
            if int(period or 1) > REGULATION_QUARTERS:
                state["home_timeouts"] = 3
                state["away_timeouts"] = 3

        atype   = str(action.get("actionType", "") or "").lower()
        subtype = str(action.get("subType",    "") or "").lower()
        poss    = action.get("possession")
        poss_id = int(poss) if poss else 0
        result  = str(action.get("shotResult", "") or "").lower()

        # ── Fouls (team bonus + foul trouble) ────────────────────────────
        if atype == "foul" and "technical" not in subtype and poss_id:
            offensive = "offensive" in subtype
            # for defensive fouls poss = fouled team → fouler is opposite
            fouling_id = poss_id if offensive else (
                away_id if poss_id == home_id else home_id
            )
            if fouling_id == home_id:
                state["home_fouls"] = state.get("home_fouls", 0) + 1
            elif fouling_id == away_id:
                state["away_fouls"] = state.get("away_fouls", 0) + 1

            # Per-player foul trouble
            pids = action.get("personIdsFilter", [])
            if pids:
                pid  = str(pids[0])
                old  = state["player_fouls"].get(pid, 0)
                new  = old + 1
                state["player_fouls"][pid] = new
                state["player_team"][pid]  = fouling_id
                if old < 4 <= new:
                    if fouling_id == home_id:
                        state["home_foul_trouble"] = state.get("home_foul_trouble", 0) + 1
                    elif fouling_id == away_id:
                        state["away_foul_trouble"] = state.get("away_foul_trouble", 0) + 1

        # ── Timeouts ─────────────────────────────────────────────────────
        elif atype == "timeout" and poss_id:
            if poss_id == home_id:
                state["home_timeouts"] = max(0, state.get("home_timeouts", 7) - 1)
            elif poss_id == away_id:
                state["away_timeouts"] = max(0, state.get("away_timeouts", 7) - 1)

        # ── Field goals (for rolling FG%) ────────────────────────────────
        elif atype in ("2pt", "3pt") and poss_id:
            # possession = shooting team for shot events
            made = result == "made"
            if poss_id == home_id:
                state["home_fga"] = state.get("home_fga", 0) + 1
                if made:
                    state["home_fgm"] = state.get("home_fgm", 0) + 1
            elif poss_id == away_id:
                state["away_fga"] = state.get("away_fga", 0) + 1
                if made:
                    state["away_fgm"] = state.get("away_fgm", 0) + 1

        # ── Momentum (last-5 scoring possessions) ────────────────────────
        sh = action.get("scoreHome")
        sa = action.get("scoreAway")
        if sh is not None and sa is not None:
            try:
                sh, sa = int(sh), int(sa)
                if sh > state.get("prev_score_home", sh):
                    state["momentum_window"].append(1)
                elif sa > state.get("prev_score_away", sa):
                    state["momentum_window"].append(-1)
                state["prev_score_home"] = sh
                state["prev_score_away"] = sa
            except (ValueError, TypeError):
                pass


def _possession_from_pbp(actions: list[dict], home_id: int) -> int:
    """Return 1 if home team has possession, 0 if away. Uses numeric team ID."""
    for action in reversed(actions):
        poss = action.get("possession")
        if poss:
            return 1 if int(poss) == home_id else 0
    return 0


def _build_game_update(scoreboard_game: dict) -> dict:
    """
    Build the full update payload for one game, merging scoreboard + live PBP.
    """
    game_id = scoreboard_game["gameId"]
    home = scoreboard_game["homeTeam"]
    away = scoreboard_game["awayTeam"]

    home_tri    = home["teamTricode"]
    away_tri    = away["teamTricode"]
    home_id     = int(home.get("teamId") or 0)
    away_id     = int(away.get("teamId") or 0)
    home_score  = int(home.get("score") or 0)
    away_score  = int(away.get("score") or 0)
    score_diff  = home_score - away_score
    period      = int(scoreboard_game.get("period") or 0)
    game_clock  = scoreboard_game.get("gameClock", "")
    game_status = int(scoreboard_game.get("gameStatus") or 1)
    seconds_left = _clock_to_seconds(game_clock, period) if period else 2880.0

    # Game is final — skip the model, return the actual result + last PBP actions
    if game_status == 3:
        home_won = score_diff > 0
        actions = _fetch_pbp(game_id)
        feed = []
        for a in actions[-10:]:
            desc = str(a.get("description", "") or "").strip()
            if not desc:
                continue
            poss = a.get("possession")
            team_label = ""
            if poss and home_id:
                team_label = home_tri if int(poss) == home_id else away_tri
            feed.append({
                "num":    int(a.get("actionNumber") or 0),
                "clock":  str(a.get("clock", "") or ""),
                "period": int(a.get("period", 0) or 0),
                "team":   team_label,
                "desc":   desc,
                "type":   str(a.get("actionType", "") or "").lower(),
                "result": str(a.get("shotResult", "") or "").lower(),
            })
        return {
            "game_id": game_id,
            "status": scoreboard_game.get("gameStatusText", "Final"),
            "game_status": 3,
            "period": period,
            "clock": game_clock,
            "home_team": home_tri,
            "away_team": away_tri,
            "home_score": home_score,
            "away_score": away_score,
            "score_diff": score_diff,
            "seconds_left": 0,
            "home_possession": 0,
            "home_in_bonus": 0,
            "away_in_bonus": 0,
            "home_fouls": 0,
            "away_fouls": 0,
            "home_win_prob": 1.0 if home_won else 0.0,
            "away_win_prob": 0.0 if home_won else 1.0,
            "feed": feed,
        }

    # Initialise per-game state on first call
    if game_id not in _game_state:
        from collections import deque
        _game_state[game_id] = {
            "period":            period,
            "home_fouls":        0,
            "away_fouls":        0,
            "home_id":           home_id,
            "away_id":           away_id,
            "seen_actions":      0,
            # new features
            "home_timeouts":     7,
            "away_timeouts":     7,
            "home_fgm":          0,
            "home_fga":          0,
            "away_fgm":          0,
            "away_fga":          0,
            "home_foul_trouble": 0,
            "away_foul_trouble": 0,
            "player_fouls":      {},
            "player_team":       {},
            "momentum_window":   deque(maxlen=5),
            "prev_score_home":   0,
            "prev_score_away":   0,
        }
    state = _game_state[game_id]
    state["home_id"] = home_id
    state["away_id"] = away_id

    # Fetch live PBP and update all tracking state
    actions = _fetch_pbp(game_id)
    _update_game_state(state, actions)

    is_ot     = period > 4
    threshold = 4 if is_ot else BONUS_THRESHOLD
    home_in_bonus   = 1 if state["away_fouls"] >= threshold else 0
    away_in_bonus   = 1 if state["home_fouls"] >= threshold else 0
    home_possession = _possession_from_pbp(actions, home_id)

    home_fg_pct = (state["home_fgm"] / state["home_fga"]
                   if state["home_fga"] else 0.0)
    away_fg_pct = (state["away_fgm"] / state["away_fga"]
                   if state["away_fga"] else 0.0)
    momentum    = sum(state["momentum_window"])

    win_prob = model_predict(
        game_id          = game_id,
        score_diff       = score_diff,
        seconds_left     = seconds_left,
        home_possession  = home_possession,
        home_in_bonus    = home_in_bonus,
        away_in_bonus    = away_in_bonus,
        home_timeouts    = state["home_timeouts"],
        away_timeouts    = state["away_timeouts"],
        home_fg_pct      = home_fg_pct,
        away_fg_pct      = away_fg_pct,
        home_foul_trouble= state["home_foul_trouble"],
        away_foul_trouble= state["away_foul_trouble"],
        momentum         = momentum,
    )

    # Last 10 events for the play-by-play feed (skip blank descriptions)
    # Live PBP uses numeric team IDs and has no teamTricode/playerNameI;
    # derive home/away label from the possession field.
    feed = []
    for a in actions[-10:]:
        desc = str(a.get("description", "") or "").strip()
        if not desc:
            continue
        poss = a.get("possession")
        team_label = ""
        if poss and home_id:
            team_label = home_tri if int(poss) == home_id else away_tri
        feed.append({
            "num":    int(a.get("actionNumber") or 0),
            "clock":  str(a.get("clock", "") or ""),
            "period": int(a.get("period", 0) or 0),
            "team":   team_label,
            "desc":   desc,
            "type":   str(a.get("actionType", "") or "").lower(),
            "result": str(a.get("shotResult", "") or "").lower(),
        })

    return {
        "game_id": game_id,
        "status": scoreboard_game.get("gameStatusText", ""),
        "game_status": game_status,
        "period": period,
        "clock": game_clock,
        "home_team": home_tri,
        "away_team": away_tri,
        "home_score": home_score,
        "away_score": away_score,
        "score_diff": score_diff,
        "seconds_left": seconds_left,
        "home_possession": home_possession,
        "home_in_bonus": home_in_bonus,
        "away_in_bonus": away_in_bonus,
        "home_fouls": state.get("home_fouls", 0),
        "away_fouls": state.get("away_fouls", 0),
        "home_win_prob": win_prob,
        "away_win_prob": round(1 - win_prob, 4),
        "feed": feed,
    }


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    return send_file(os.path.abspath(path))


@app.route("/games")
def games():
    raw = _fetch_scoreboard()
    return jsonify([
        {
            "game_id": g["gameId"],
            "status":    g["gameStatusText"],
            "period":    g.get("period", 0),
            "clock":     g.get("gameClock", ""),
            "home_team": g["homeTeam"]["teamTricode"],
            "away_team": g["awayTeam"]["teamTricode"],
            "home_score": g["homeTeam"].get("score", 0),
            "away_score": g["awayTeam"].get("score", 0),
        }
        for g in raw
    ])


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    body = request.get_json(force=True)
    required = ["score_diff", "seconds_left", "home_possession", "home_in_bonus", "away_in_bonus"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    prob = model_predict(
        float(body["score_diff"]), float(body["seconds_left"]),
        int(body["home_possession"]), int(body["home_in_bonus"]), int(body["away_in_bonus"]),
    )
    return jsonify({"home_win_prob": prob, "away_win_prob": round(1 - prob, 4)})


@app.route("/boxscore/<game_id>")
def get_boxscore(game_id):
    url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
    data = _get(url)
    if not data:
        return jsonify({"error": "Boxscore unavailable"}), 404

    game = data.get("game", {})

    def parse_team(team):
        players = []
        for p in team.get("players", []):
            if p.get("played", "0") != "1":
                continue
            s = p.get("statistics", {})
            mins_raw = s.get("minutesCalculated", "") or ""
            m = re.match(r"PT(\d+)M", mins_raw)
            mins = int(m.group(1)) if m else 0
            players.append({
                "name":    p.get("nameI", ""),
                "starter": p.get("starter", "0") == "1",
                "mins":    mins,
                "pts":     s.get("points", 0),
                "reb":     s.get("reboundsTotal", 0),
                "ast":     s.get("assists", 0),
                "stl":     s.get("steals", 0),
                "blk":     s.get("blocks", 0),
                "fgm":     s.get("fieldGoalsMade", 0),
                "fga":     s.get("fieldGoalsAttempted", 0),
            })
        return {
            "tricode": team.get("teamTricode", ""),
            "players": sorted(players, key=lambda x: (-x["starter"], -x["pts"])),
        }

    return jsonify({
        "home": parse_team(game.get("homeTeam", {})),
        "away": parse_team(game.get("awayTeam", {})),
    })


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

def _poll_loop() -> None:
    while True:
        time.sleep(POLL_INTERVAL)

        with _sub_lock:
            watched = dict(_subscriptions)  # snapshot

        if not watched:
            continue

        games = _fetch_scoreboard()
        game_map = {g["gameId"]: g for g in games}

        for game_id, sids in watched.items():
            if not sids:
                continue
            if game_id not in game_map:
                socketio.emit("error", {"message": f"Game {game_id} not on today's scoreboard"}, to=list(sids)[0])
                continue
            try:
                payload = _build_game_update(game_map[game_id])
                for sid in list(sids):
                    socketio.emit("game_update", payload, to=sid)
            except Exception as exc:
                print(f"[poll] error for {game_id}: {exc}")


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    print(f"[ws] client connected: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    with _sub_lock:
        for sids in _subscriptions.values():
            sids.discard(sid)
    print(f"[ws] client disconnected: {sid}")


@socketio.on("subscribe")
def on_subscribe(data):
    game_id = str(data.get("game_id", ""))
    if not game_id:
        emit("error", {"message": "game_id required"})
        return
    with _sub_lock:
        _subscriptions.setdefault(game_id, set()).add(request.sid)
    print(f"[ws] {request.sid} subscribed to {game_id}")
    emit("subscribed", {"game_id": game_id})


@socketio.on("unsubscribe")
def on_unsubscribe(data):
    game_id = str(data.get("game_id", ""))
    with _sub_lock:
        if game_id in _subscriptions:
            _subscriptions[game_id].discard(request.sid)
    emit("unsubscribed", {"game_id": game_id})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print("Server running on http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
