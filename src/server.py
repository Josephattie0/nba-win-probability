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

def _update_foul_state(state: dict, actions: list[dict]) -> None:
    """
    Incrementally update per-game foul state from new PBP actions.

    The live PBP uses numeric team IDs in the `possession` field, not tricodes.
    For defensive fouls (personal/shooting/loose ball), possession after the
    foul goes to the fouled team — so the fouling team is the one WITHOUT
    possession. Offensive fouls are the opposite but rare; we accept that error.
    """
    seen = state.get("seen_actions", 0)
    new_actions = actions[seen:]
    state["seen_actions"] = len(actions)

    home_id = state.get("home_id", 0)
    away_id = state.get("away_id", 0)

    for action in new_actions:
        period = action.get("period", state.get("period", 1))

        if period != state.get("period"):
            state["period"] = period
            state["home_fouls"] = 0
            state["away_fouls"] = 0

        if str(action.get("actionType", "")).lower() != "foul":
            continue

        subtype = str(action.get("subType", "")).lower()
        poss = action.get("possession")
        if not poss or not home_id:
            continue

        poss = int(poss)
        offensive = "offensive" in subtype
        if offensive:
            # Offensive foul: fouling team HAS possession
            if poss == home_id:
                state["home_fouls"] = state.get("home_fouls", 0) + 1
            elif poss == away_id:
                state["away_fouls"] = state.get("away_fouls", 0) + 1
        else:
            # Defensive foul: fouling team is opposite of possession after foul
            if poss == home_id:
                state["away_fouls"] = state.get("away_fouls", 0) + 1
            elif poss == away_id:
                state["home_fouls"] = state.get("home_fouls", 0) + 1


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

    home_tri = home["teamTricode"]
    away_tri = away["teamTricode"]
    home_id  = int(home.get("teamId") or 0)
    away_id  = int(away.get("teamId") or 0)
    home_score = int(home.get("score") or 0)
    away_score = int(away.get("score") or 0)
    score_diff = home_score - away_score
    period = int(scoreboard_game.get("period") or 0)
    game_clock = scoreboard_game.get("gameClock", "")
    seconds_left = _clock_to_seconds(game_clock, period) if period else 2880.0

    # Initialize per-game foul state on first call
    if game_id not in _game_state:
        _game_state[game_id] = {
            "period": period,
            "home_fouls": 0,
            "away_fouls": 0,
            "home_id": home_id,
            "away_id": away_id,
            "seen_actions": 0,
        }
    state = _game_state[game_id]
    state["home_id"] = home_id
    state["away_id"] = away_id

    # Fetch live PBP for possession + bonus (may be empty before game starts)
    actions = _fetch_pbp(game_id)
    _update_foul_state(state, actions)

    is_ot = period > 4
    threshold = 4 if is_ot else BONUS_THRESHOLD
    home_in_bonus = 1 if state["away_fouls"] >= threshold else 0
    away_in_bonus = 1 if state["home_fouls"] >= threshold else 0
    home_possession = _possession_from_pbp(actions, home_id)

    win_prob = model_predict(score_diff, seconds_left, home_possession, home_in_bonus, away_in_bonus)

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
# REST endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    return send_file(os.path.abspath(path))


@app.route("/games")
def games():
    """List today's games with their current status."""
    raw = _fetch_scoreboard()
    return jsonify([
        {
            "game_id": g["gameId"],
            "status": g["gameStatusText"],
            "period": g.get("period", 0),
            "clock": g.get("gameClock", ""),
            "home_team": g["homeTeam"]["teamTricode"],
            "away_team": g["awayTeam"]["teamTricode"],
            "home_score": g["homeTeam"].get("score", 0),
            "away_score": g["awayTeam"].get("score", 0),
        }
        for g in raw
    ])


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    One-shot prediction.

    Body (JSON):
        score_diff      int    home minus away points
        seconds_left    float  total seconds remaining
        home_possession int    1 or 0
        home_in_bonus   int    1 or 0
        away_in_bonus   int    1 or 0
    """
    body = request.get_json(force=True)
    required = ["score_diff", "seconds_left", "home_possession", "home_in_bonus", "away_in_bonus"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    prob = model_predict(
        float(body["score_diff"]),
        float(body["seconds_left"]),
        int(body["home_possession"]),
        int(body["home_in_bonus"]),
        int(body["away_in_bonus"]),
    )
    return jsonify({"home_win_prob": prob, "away_win_prob": round(1 - prob, 4)})


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
