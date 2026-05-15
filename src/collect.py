"""
Phase 1 — Data collection.

Pulls PlayByPlayV3 for every game in the given seasons, computes game-state
features at each event, labels each row with the final outcome, and writes the
result to ../data/play_by_play.csv.

Usage:
    python collect.py                     # defaults: 2022-23, 2023-24
    python collect.py --seasons 2021-22   # one season
    python collect.py --max-games 50      # quick smoke-test run
"""

import argparse
import os
import time

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, PlayByPlayV3

from features import build_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "play_by_play.csv")

# nba_api is rate-limited; sleep between requests to avoid 429s
REQUEST_DELAY = 0.7  # seconds


def get_game_ids(seasons: list[str]) -> list[str]:
    """Return unique game IDs for the given seasons (regular season only)."""
    ids = []
    for season in seasons:
        log = LeagueGameLog(season=season, season_type_all_star="Regular Season")
        df = log.get_data_frames()[0]
        ids.extend(df["GAME_ID"].unique().tolist())
    return list(dict.fromkeys(ids))  # deduplicate, preserve order


def fetch_pbp(game_id: str) -> pd.DataFrame | None:
    """Fetch raw play-by-play for one game; return None on failure."""
    try:
        pbp = PlayByPlayV3(game_id=game_id)
        df = pbp.get_data_frames()[0]
        if df.empty:
            return None
        df["GAME_ID"] = game_id
        return df
    except Exception as exc:
        print(f"  [warn] {game_id}: {exc}")
        return None


def collect(seasons: list[str], max_games: int | None = None) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching game IDs for seasons: {seasons}")
    game_ids = get_game_ids(seasons)
    if max_games:
        game_ids = game_ids[:max_games]
    print(f"Total games to process: {len(game_ids)}")

    all_rows: list[pd.DataFrame] = []

    for i, gid in enumerate(game_ids, 1):
        print(f"[{i}/{len(game_ids)}] {gid}", end=" ... ", flush=True)
        raw = fetch_pbp(gid)
        time.sleep(REQUEST_DELAY)

        if raw is None:
            print("skipped")
            continue

        try:
            features = build_features(raw)
            all_rows.append(features)
            print(f"ok ({len(features)} rows)")
        except Exception as exc:
            print(f"feature error: {exc}")

    if not all_rows:
        raise RuntimeError("No data collected — check your nba_api connection.")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(combined):,} rows to {OUT_PATH}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        help="NBA season strings, e.g. 2022-23",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Cap the number of games (useful for testing)",
    )
    args = parser.parse_args()
    collect(args.seasons, args.max_games)
