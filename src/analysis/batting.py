"""Batting analysis using SQLite queries and pandas."""

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
    for name, query in queries.items():
        results[name] = pd.read_sql_query(query, conn)
    return results


def compute_career_trajectory(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute runs per season for each batsman."""
    query = (
        "SELECT batsman, season, SUM(batsman_runs) AS runs "
        "FROM deliveries GROUP BY batsman, season"
    )
    return pd.read_sql_query(query, conn)


def compute_head_to_head(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute a strike-rate matrix for top batsmen vs top bowlers."""
    deliveries = pd.read_sql_query(
        "SELECT batsman, bowler, batsman_runs, wide_runs, dismissal_kind, is_wicket "
        "FROM deliveries",
        conn,
    )
    deliveries["legal_ball"] = (deliveries["wide_runs"].fillna(0) == 0).astype(int)

    top_batsmen = (
        deliveries.groupby("batsman")["batsman_runs"].sum().nlargest(10).index
    )

    wicket_mask = deliveries["is_wicket"].fillna(0).astype(int) == 1
    dismissal_mask = ~deliveries["dismissal_kind"].fillna("").isin(
        ["run out", "retired hurt", "retired out", "obstructing the field"]
    )
    top_bowlers = (
        deliveries[wicket_mask & dismissal_mask]
        .groupby("bowler")["is_wicket"]
        .count()
        .nlargest(10)
        .index
    )

    filtered = deliveries[
        deliveries["batsman"].isin(top_batsmen)
        & deliveries["bowler"].isin(top_bowlers)
    ].copy()

    grouped = (
        filtered.groupby(["batsman", "bowler"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    grouped["strike_rate"] = 100.0 * grouped["runs"] / grouped["balls"].replace(0, pd.NA)

    return grouped.pivot(index="batsman", columns="bowler", values="strike_rate")


def main() -> None:
    """Run batting SQL analysis and additional metrics."""
    project_root = get_project_root()
    db_path = project_root / "data" / "ipl.db"
    sql_path = project_root / "sql" / "batting_analysis.sql"

    with sqlite3.connect(db_path) as conn:
        queries = load_named_queries(sql_path)
        results = run_queries(conn, queries)

        print("\n=== Batting SQL Results ===")
        for name, df in results.items():
            print(f"\n{name}")
            print(df.head(10))

        career = compute_career_trajectory(conn)
        print("\nCareer Trajectory (sample)")
        print(career.head(10))

        h2h = compute_head_to_head(conn)
        print("\nHead-to-Head Strike Rate Matrix (sample)")
        print(h2h.head(10))


if __name__ == "__main__":
    main()
