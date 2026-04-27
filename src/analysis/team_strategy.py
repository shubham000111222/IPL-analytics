"""Team strategy analysis using SQLite queries and pandas."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


TEAM_HOME_CITIES = {
    "Chennai Super Kings": ["Chennai"],
    "Mumbai Indians": ["Mumbai"],
    "Royal Challengers Bangalore": ["Bengaluru", "Bangalore"],
    "Kolkata Knight Riders": ["Kolkata"],
    "Delhi Capitals": ["Delhi"],
    "Rajasthan Royals": ["Jaipur"],
    "Sunrisers Hyderabad": ["Hyderabad"],
    "Punjab Kings": ["Mohali", "Chandigarh", "Dharamsala"],
    "Gujarat Titans": ["Ahmedabad"],
    "Lucknow Super Giants": ["Lucknow"],
}


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_named_queries(path: Path) -> Dict[str, str]:
    """Load named SQL queries from a file using -- name: markers."""
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("-- name:") if block.strip()]
    queries: Dict[str, str] = {}
    for block in blocks:
        lines = block.splitlines()
        name = lines[0].strip()
        sql = "\n".join(lines[1:]).strip()
        if sql.endswith(";"):
            sql = sql[:-1]
        queries[name] = sql
    return queries


def run_queries(conn: sqlite3.Connection, queries: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Run SQL queries and return results as DataFrames."""
    results: Dict[str, pd.DataFrame] = {}
    for name, query in queries.items():
        results[name] = pd.read_sql_query(query, conn)
    return results


def compute_clutch_performance(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute team performance in last 5 overs when required rate exceeds 10."""
    deliveries = pd.read_sql_query(
        "SELECT match_id, inning, over, batting_team, total_runs, is_wicket "
        "FROM deliveries",
        conn,
    )
    first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("target")
    )

    second_innings = deliveries[deliveries["inning"] == 2].copy()
    over_runs = (
        second_innings.groupby(["match_id", "batting_team", "over"])["total_runs"]
        .sum()
        .reset_index()
    )
    over_runs = over_runs.sort_values(["match_id", "over"])
    over_runs["runs_scored"] = over_runs.groupby("match_id")["total_runs"].cumsum()
    over_runs = over_runs.merge(first_innings, on="match_id", how="left")
    over_runs["overs_left"] = 20 - over_runs["over"]
    over_runs["required_rate"] = (
        (over_runs["target"] + 1 - over_runs["runs_scored"]) / over_runs["overs_left"].replace(0, pd.NA)
    )

    clutch_overs = over_runs[
        (over_runs["over"] >= 16)
        & (over_runs["required_rate"] > 10)
    ].copy()

    wickets = (
        second_innings[second_innings["over"] >= 16]
        .groupby(["match_id", "batting_team"])["is_wicket"]
        .sum()
        .reset_index()
        .rename(columns={"is_wicket": "wickets_lost"})
    )

    clutch_agg = (
        clutch_overs.groupby(["match_id", "batting_team"])
        .agg(runs_scored=("total_runs", "sum"), overs=("over", "nunique"))
        .reset_index()
    )
    clutch_agg = clutch_agg.merge(wickets, on=["match_id", "batting_team"], how="left")

    matches = pd.read_sql_query("SELECT match_id, winner FROM matches", conn)
    clutch_agg = clutch_agg.merge(matches, on="match_id", how="left")
    clutch_agg["win"] = (clutch_agg["batting_team"] == clutch_agg["winner"]).astype(int)
    clutch_agg["runs_per_over"] = clutch_agg["runs_scored"] / clutch_agg["overs"].replace(0, pd.NA)

    return (
        clutch_agg.groupby("batting_team")
        .agg(
            clutch_matches=("match_id", "nunique"),
            avg_runs_per_over=("runs_per_over", "mean"),
            avg_wickets_lost=("wickets_lost", "mean"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .sort_values("win_rate", ascending=False)
    )


def compute_home_away_win_rate(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute home vs away win rates per team using city as proxy for home venues."""
    matches = pd.read_sql_query(
        "SELECT match_id, season, team1, team2, winner, city FROM matches",
        conn,
    )

    team_rows = pd.concat(
        [
            matches[["match_id", "team1", "winner", "city"]].rename(columns={"team1": "team"}),
            matches[["match_id", "team2", "winner", "city"]].rename(columns={"team2": "team"}),
        ],
        ignore_index=True,
    )

    def is_home(row: pd.Series) -> bool:
        city = str(row.get("city", ""))
        return city in TEAM_HOME_CITIES.get(row["team"], [])

    team_rows["home_away"] = team_rows.apply(
        lambda row: "home" if is_home(row) else "away", axis=1
    )
    team_rows["win"] = (team_rows["team"] == team_rows["winner"]).astype(int)

    return (
        team_rows.groupby(["team", "home_away"])
        .agg(matches=("match_id", "nunique"), wins=("win", "sum"))
        .reset_index()
        .assign(win_rate=lambda df: 100.0 * df["wins"] / df["matches"].replace(0, pd.NA))
        .sort_values(["team", "home_away"])
    )


def main() -> None:
    """Run team strategy SQL analysis and additional metrics."""
    project_root = get_project_root()
    db_path = project_root / "data" / "ipl.db"
    sql_path = project_root / "sql" / "team_strategy.sql"

    with sqlite3.connect(db_path) as conn:
        queries = load_named_queries(sql_path)
        results = run_queries(conn, queries)

        print("\n=== Team Strategy SQL Results ===")
        for name, df in results.items():
            print(f"\n{name}")
            print(df.head(10))

        clutch = compute_clutch_performance(conn)
        print("\nClutch Performance (sample)")
        print(clutch.head(10))

        home_away = compute_home_away_win_rate(conn)
        print("\nHome vs Away Win Rate (sample)")
        print(home_away.head(10))


if __name__ == "__main__":
    main()
