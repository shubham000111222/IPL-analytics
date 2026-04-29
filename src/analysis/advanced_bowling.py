"""Advanced bowling analysis for IPL analytics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


WICKET_EXCLUSIONS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection."""
    if db_path is None:
        db_path = get_project_root() / "data" / "ipl.db"
    return sqlite3.connect(db_path)


def bowler_batsman_matchup_matrix(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return a batsman vs bowler economy matrix for top players."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, batsman, bowler, batsman_runs, wide_runs, noball_runs, "
            "dismissal_kind, player_dismissed, is_wicket "
            "FROM deliveries",
            conn,
        )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)

    top_batsmen = (
        deliveries.groupby("batsman")["batsman_runs"].sum().nlargest(15).index
    )

    wicket_mask = (
        (deliveries["is_wicket"] == 1)
        & (~deliveries["dismissal_kind"].fillna("").str.lower().isin(WICKET_EXCLUSIONS))
    )
    top_bowlers = (
        deliveries[wicket_mask].groupby("bowler")["is_wicket"].sum().nlargest(15).index
    )

    filtered = deliveries[
        deliveries["batsman"].isin(top_batsmen)
        & deliveries["bowler"].isin(top_bowlers)
    ].copy()

    filtered["dismissal"] = (
        (filtered["player_dismissed"] == filtered["batsman"]) & wicket_mask
    ).astype(int)

    agg = (
        filtered.groupby(["batsman", "bowler"])
        .agg(
            balls=("legal_ball", "sum"),
            runs=("batsman_runs", "sum"),
            dismissals=("dismissal", "sum"),
        )
        .reset_index()
    )
    agg["economy"] = 6.0 * agg["runs"] / agg["balls"].replace(0, pd.NA)

    return agg.pivot(index="batsman", columns="bowler", values="economy")


def _dots_before_wicket(group: pd.DataFrame) -> pd.DataFrame:
    """Return dot counts before each wicket for a bowler/innings group."""
    group = group.sort_values(["over", "ball"]).copy()
    streak = 0
    dots_before = []
    for _, row in group.iterrows():
        if row["is_wicket"] == 1 and row["is_bowler_wicket"] == 1:
            dots_before.append(streak)
        if row["dot_ball"] == 1:
            streak += 1
        else:
            streak = 0
    if not dots_before:
        return pd.DataFrame(columns=["bowler", "dots_before_wicket"])
    if "bowler" in group.columns:
        bowler_name = group["bowler"].iloc[0]
    else:
        bowler_name = group.name[2] if isinstance(group.name, tuple) else group.name
    return pd.DataFrame(
        {
            "bowler": [bowler_name] * len(dots_before),
            "dots_before_wicket": dots_before,
        }
    )


def pressure_index(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return average consecutive dots before wickets and dot-ball pct per bowler."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, over, ball, bowler, total_runs, wide_runs, noball_runs, "
            "dismissal_kind, is_wicket "
            "FROM deliveries",
            conn,
        )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)
    deliveries = deliveries[deliveries["legal_ball"] == 1].copy()
    deliveries["dot_ball"] = (deliveries["total_runs"].fillna(0) == 0).astype(int)
    deliveries["is_bowler_wicket"] = (
        (deliveries["is_wicket"] == 1)
        & (~deliveries["dismissal_kind"].fillna("").str.lower().isin(WICKET_EXCLUSIONS))
    ).astype(int)

    dot_records = []
    for (_, _, bowler), group in deliveries.groupby(["match_id", "inning", "bowler"]):
        group = group.sort_values(["over", "ball"])
        streak = 0
        for _, row in group.iterrows():
            if row["is_bowler_wicket"] == 1:
                dot_records.append({"bowler": bowler, "dots_before_wicket": streak})
            if row["dot_ball"] == 1:
                streak += 1
            else:
                streak = 0
    dots = pd.DataFrame(dot_records)

    dot_pct = (
        deliveries.groupby("bowler")
        .agg(dot_balls=("dot_ball", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    dot_pct["dot_ball_pct"] = 100.0 * dot_pct["dot_balls"] / dot_pct["balls"].replace(0, pd.NA)

    agg = (
        dots.groupby("bowler")["dots_before_wicket"]
        .mean()
        .reset_index()
        .rename(columns={"dots_before_wicket": "avg_dots_before_wicket"})
    )
    wicket_counts = (
        dots.groupby("bowler")["dots_before_wicket"]
        .count()
        .reset_index()
        .rename(columns={"dots_before_wicket": "wickets"})
    )
    agg = agg.merge(wicket_counts, on="bowler", how="left")
    agg = agg.merge(dot_pct[["bowler", "dot_ball_pct", "balls"]], on="bowler", how="left")
    agg = agg[(agg["balls"] >= 200) & (agg["wickets"] >= 30)]
    agg = agg.sort_values("avg_dots_before_wicket", ascending=False)

    return agg


def wicket_timing_heatmap_data(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return wickets by over for the top 20 bowlers."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT bowler, over, dismissal_kind, is_wicket "
            "FROM deliveries",
            conn,
        )

    wickets = deliveries[
        (deliveries["is_wicket"] == 1)
        & (~deliveries["dismissal_kind"].fillna("").str.lower().isin(WICKET_EXCLUSIONS))
    ].copy()

    top_bowlers = wickets.groupby("bowler")["is_wicket"].sum().nlargest(20).index
    wickets = wickets[wickets["bowler"].isin(top_bowlers)]

    agg = (
        wickets.groupby(["bowler", "over"])
        .size()
        .reset_index(name="wickets")
    )

    return agg.pivot(index="bowler", columns="over", values="wickets").fillna(0)


def discipline_score(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return wide/no-ball rate per bowler (min 500 balls)."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT bowler, wide_runs, noball_runs "
            "FROM deliveries",
            conn,
        )

    deliveries["wide_ball"] = (deliveries["wide_runs"].fillna(0) > 0).astype(int)
    deliveries["noball_ball"] = (deliveries["noball_runs"].fillna(0) > 0).astype(int)

    agg = (
        deliveries.groupby("bowler")
        .agg(
            wide_balls=("wide_ball", "sum"),
            noballs=("noball_ball", "sum"),
            total_balls=("bowler", "count"),
        )
        .reset_index()
    )
    agg = agg[agg["total_balls"] >= 500]
    agg["discipline_score"] = 100.0 * (agg["wide_balls"] + agg["noballs"]) / agg[
        "total_balls"
    ].replace(0, pd.NA)

    return agg.sort_values("discipline_score")


def main() -> None:
    """Run advanced bowling analysis functions and print samples."""
    matchup = bowler_batsman_matchup_matrix()
    print("\nBowler vs Batsman Matchup (sample)")
    print(matchup.head(5))

    pressure = pressure_index()
    print("\nPressure Index (sample)")
    print(pressure.head(10))

    timing = wicket_timing_heatmap_data()
    print("\nWicket Timing (sample)")
    print(timing.head(10))

    discipline = discipline_score()
    print("\nDiscipline Score (sample)")
    print(discipline.head(10))


if __name__ == "__main__":
    main()
