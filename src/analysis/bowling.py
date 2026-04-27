"""Bowling analysis using SQLite queries and pandas."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd


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


def fetch_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Fetch the column names for a SQLite table."""
    info = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [row[1] for row in info]


def run_queries(conn: sqlite3.Connection, queries: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Run SQL queries and return results as DataFrames."""
    results: Dict[str, pd.DataFrame] = {}
    columns = fetch_table_columns(conn, "deliveries")

    for name, query in queries.items():
        if "Left/Right" in name and "batsman_hand" not in columns:
            print("\nSkipping left/right handedness query (missing batsman_hand column).")
            continue
        results[name] = pd.read_sql_query(query, conn)
    return results


def compute_wicket_type_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute wicket type distribution per bowler."""
    query = (
        "SELECT bowler, dismissal_kind, COUNT(*) AS wickets "
        "FROM deliveries "
        "WHERE is_wicket = 1 AND dismissal_kind IS NOT NULL "
        "GROUP BY bowler, dismissal_kind"
    )
    return pd.read_sql_query(query, conn)


def compute_economy_pressure_index(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute economy pressure index per bowler."""
    deliveries = pd.read_sql_query(
        "SELECT bowler, total_runs, wide_runs, noball_runs, bye_runs, legbye_runs, penalty_runs "
        "FROM deliveries",
        conn,
    )
    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)
    deliveries["runs_conceded"] = (
        deliveries["total_runs"].fillna(0)
        - deliveries["bye_runs"].fillna(0)
        - deliveries["legbye_runs"].fillna(0)
        - deliveries["penalty_runs"].fillna(0)
    )
    deliveries["dot_ball"] = (deliveries["total_runs"].fillna(0) == 0).astype(int)

    agg = (
        deliveries.groupby("bowler")
        .agg(
            runs_conceded=("runs_conceded", "sum"),
            legal_balls=("legal_ball", "sum"),
            dot_balls=("dot_ball", "sum"),
        )
        .reset_index()
    )

    agg = agg[agg["legal_balls"] >= 200]
    agg["economy"] = 6.0 * agg["runs_conceded"] / agg["legal_balls"].replace(0, pd.NA)
    agg["dot_ball_pct"] = 100.0 * agg["dot_balls"] / agg["legal_balls"].replace(0, pd.NA)
    agg["economy_pressure_index"] = agg["economy"] * (1 - agg["dot_ball_pct"] / 100.0)

    return agg.sort_values("economy_pressure_index")


def main() -> None:
    """Run bowling SQL analysis and additional metrics."""
    project_root = get_project_root()
    db_path = project_root / "data" / "ipl.db"
    sql_path = project_root / "sql" / "bowling_analysis.sql"

    with sqlite3.connect(db_path) as conn:
        queries = load_named_queries(sql_path)
        results = run_queries(conn, queries)

        print("\n=== Bowling SQL Results ===")
        for name, df in results.items():
            print(f"\n{name}")
            print(df.head(10))

        wicket_dist = compute_wicket_type_distribution(conn)
        print("\nWicket Type Distribution (sample)")
        print(wicket_dist.head(10))

        pressure = compute_economy_pressure_index(conn)
        print("\nEconomy Pressure Index (sample)")
        print(pressure.head(10))


if __name__ == "__main__":
    main()
