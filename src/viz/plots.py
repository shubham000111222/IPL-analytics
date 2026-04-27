"""Create publication-quality IPL analytics plots."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLOR_PALETTE = [
    "#003f5c",
    "#2f4b7c",
    "#665191",
    "#a05195",
    "#d45087",
    "#f95d6a",
    "#ff7c43",
    "#ffa600",
]


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def apply_plot_style() -> None:
    """Apply a clean custom matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "axes.facecolor": "#fbfbfb",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )
    sns.set_palette(COLOR_PALETTE)


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save a matplotlib figure to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fetch_dataframe(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Fetch a DataFrame from a SQLite query."""
    return pd.read_sql_query(query, conn)


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Return column names for a SQLite table."""
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [row[1] for row in rows]


def build_team_color_map(teams: List[str]) -> Dict[str, str]:
    """Assign consistent colors to teams from a shared palette."""
    colors = {}
    for idx, team in enumerate(sorted(teams)):
        colors[team] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    return colors


def plot_top_run_scorers(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot top 15 run scorers with team colors."""
    query = (
        "WITH batsman_team_runs AS ("
        "  SELECT batsman, batting_team, SUM(batsman_runs) AS runs "
        "  FROM deliveries GROUP BY batsman, batting_team"
        "), ranked AS ("
        "  SELECT batsman, batting_team, runs, "
        "         ROW_NUMBER() OVER (PARTITION BY batsman ORDER BY runs DESC) AS rn "
        "  FROM batsman_team_runs"
        "), totals AS ("
        "  SELECT batsman, SUM(runs) AS total_runs FROM batsman_team_runs GROUP BY batsman"
        ")"
        " SELECT t.batsman, t.total_runs, r.batting_team AS team "
        " FROM totals t JOIN ranked r ON r.batsman = t.batsman AND r.rn = 1 "
        " ORDER BY total_runs DESC LIMIT 15"
    )
    df = fetch_dataframe(conn, query)
    team_colors = build_team_color_map(df["team"].unique().tolist())

    fig, ax = plt.subplots()
    ax.barh(df["batsman"], df["total_runs"], color=[team_colors[t] for t in df["team"]])
    ax.set_title("Top 15 Run Scorers All Time")
    ax.set_xlabel("Runs")
    ax.set_ylabel("Batsman")
    ax.invert_yaxis()

    save_plot(fig, output_dir / "top_run_scorers.png")


def plot_top_wicket_takers(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot top 15 wicket takers all time."""
    query = (
        "SELECT bowler, "
        "SUM(CASE WHEN is_wicket = 1 AND COALESCE(dismissal_kind, '') NOT IN "
        "('run out', 'retired hurt', 'retired out', 'obstructing the field') "
        "THEN 1 ELSE 0 END) AS wickets "
        "FROM deliveries GROUP BY bowler ORDER BY wickets DESC LIMIT 15"
    )
    df = fetch_dataframe(conn, query)

    fig, ax = plt.subplots()
    ax.barh(df["bowler"], df["wickets"], color=COLOR_PALETTE[1])
    ax.set_title("Top 15 Wicket Takers All Time")
    ax.set_xlabel("Wickets")
    ax.set_ylabel("Bowler")
    ax.invert_yaxis()

    save_plot(fig, output_dir / "top_wicket_takers.png")


def plot_strike_rate_phase_heatmap(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot strike rate by phase heatmap for top batsmen."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT batsman, over, batsman_runs, wide_runs FROM deliveries",
    )
    deliveries["legal_ball"] = (deliveries["wide_runs"].fillna(0) == 0).astype(int)

    balls = (
        deliveries.groupby("batsman")["legal_ball"].sum().sort_values(ascending=False)
    )
    top_batsmen = balls.head(20).index
    filtered = deliveries[deliveries["batsman"].isin(top_batsmen)].copy()

    phase_bins = pd.cut(
        filtered["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    )
    filtered["phase"] = phase_bins

    agg = (
        filtered.groupby(["batsman", "phase"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    agg["strike_rate"] = 100.0 * agg["runs"] / agg["balls"].replace(0, pd.NA)

    pivot = agg.pivot(index="batsman", columns="phase", values="strike_rate").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False, ax=ax)
    ax.set_title("Strike Rate by Phase (Top 20 Batsmen)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Batsman")

    save_plot(fig, output_dir / "strike_rate_phase_heatmap.png")


def plot_economy_phase_heatmap(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot economy rate by phase heatmap for top bowlers."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT bowler, over, total_runs, wide_runs, noball_runs, bye_runs, legbye_runs, penalty_runs "
        "FROM deliveries",
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

    balls = deliveries.groupby("bowler")["legal_ball"].sum().sort_values(ascending=False)
    top_bowlers = balls.head(20).index
    filtered = deliveries[deliveries["bowler"].isin(top_bowlers)].copy()

    phase_bins = pd.cut(
        filtered["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    )
    filtered["phase"] = phase_bins

    agg = (
        filtered.groupby(["bowler", "phase"])
        .agg(runs=("runs_conceded", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    agg["economy"] = 6.0 * agg["runs"] / agg["balls"].replace(0, pd.NA)
    pivot = agg.pivot(index="bowler", columns="phase", values="economy").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="Reds", annot=False, ax=ax)
    ax.set_title("Economy Rate by Phase (Top 20 Bowlers)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Bowler")

    save_plot(fig, output_dir / "economy_phase_heatmap.png")


def plot_toss_win_match_win(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot match win rate for toss winners by team and decision."""
    matches = fetch_dataframe(
        conn,
        "SELECT toss_winner, toss_decision, winner FROM matches",
    )
    matches["win"] = (matches["toss_winner"] == matches["winner"]).astype(int)
    agg = (
        matches.groupby(["toss_winner", "toss_decision"])
        .agg(win_rate=("win", "mean"))
        .reset_index()
    )
    agg["win_rate"] = agg["win_rate"] * 100.0

    pivot = agg.pivot(index="toss_winner", columns="toss_decision", values="win_rate").fillna(0)
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, color=COLOR_PALETTE[: len(pivot.columns)])
    ax.set_title("Toss Win to Match Win Rate")
    ax.set_xlabel("Team")
    ax.set_ylabel("Win Rate (%)")
    ax.legend(title="Toss Decision")

    save_plot(fig, output_dir / "toss_win_match_win.png")


def plot_venue_run_rates(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot venue run rates vs chasing success."""
    query = (
        "WITH innings_totals AS ("
        "  SELECT match_id, inning, batting_team, SUM(total_runs) AS runs "
        "  FROM deliveries GROUP BY match_id, inning, batting_team"
        "), first_innings AS ("
        "  SELECT match_id, runs AS first_innings_runs FROM innings_totals WHERE inning = 1"
        "), chase_success AS ("
        "  SELECT m.match_id, CASE WHEN m.winner = i2.batting_team THEN 1 ELSE 0 END AS chase_win "
        "  FROM matches m LEFT JOIN innings_totals i2 ON i2.match_id = m.match_id AND i2.inning = 2"
        ")"
        " SELECT m.venue, COUNT(*) AS matches, AVG(f.first_innings_runs) AS avg_first_innings_score, "
        " 100.0 * SUM(c.chase_win) / NULLIF(COUNT(*), 0) AS chasing_success_rate "
        " FROM matches m "
        " LEFT JOIN first_innings f ON f.match_id = m.match_id "
        " LEFT JOIN chase_success c ON c.match_id = m.match_id "
        " GROUP BY m.venue HAVING COUNT(*) >= 10"
    )
    df = fetch_dataframe(conn, query)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(df["avg_first_innings_score"], df["chasing_success_rate"], color=COLOR_PALETTE[2])
    for _, row in df.iterrows():
        ax.annotate(row["venue"], (row["avg_first_innings_score"], row["chasing_success_rate"]))
    ax.set_title("Venue Run Rates vs Chasing Success")
    ax.set_xlabel("Average First Innings Score")
    ax.set_ylabel("Chasing Success Rate (%)")

    save_plot(fig, output_dir / "venue_run_rates.png")


def plot_season_trends(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot season trends in runs, wickets, and boundary percentage."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, season, total_runs, batsman_runs, is_wicket FROM deliveries",
    )

    match_totals = (
        deliveries.groupby(["season", "match_id"])
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            boundary_runs=(
                "batsman_runs",
                lambda x: (x == 4).sum() * 4 + (x == 6).sum() * 6,
            ),
        )
        .reset_index()
    )

    season_totals = (
        match_totals.groupby("season")
        .agg(
            avg_runs_per_match=("runs", "mean"),
            avg_wickets_per_match=("wickets", "mean"),
            runs_sum=("runs", "sum"),
            boundary_runs_sum=("boundary_runs", "sum"),
        )
        .reset_index()
    )
    season_totals["boundary_pct"] = (
        100.0
        * season_totals["boundary_runs_sum"]
        / season_totals["runs_sum"].replace(0, pd.NA)
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        season_totals["season"],
        season_totals["avg_runs_per_match"],
        marker="o",
        color=COLOR_PALETTE[0],
        label="Avg Runs per Match",
    )
    ax1.plot(
        season_totals["season"],
        season_totals["avg_wickets_per_match"],
        marker="o",
        color=COLOR_PALETTE[1],
        label="Avg Wickets per Match",
    )
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Runs / Wickets")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        season_totals["season"],
        season_totals["boundary_pct"],
        marker="o",
        color=COLOR_PALETTE[3],
        label="Boundary %",
    )
    ax2.set_ylabel("Boundary %")

    ax1.set_title("Season Trends in Runs, Wickets, and Boundary %")

    save_plot(fig, output_dir / "season_trends.png")


def plot_dynasty_chart(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot IPL dynasty chart of titles per team per season."""
    columns = get_table_columns(conn, "matches")
    select_cols = ["match_id", "season", "winner"]
    if "date" in columns:
        select_cols.append("date")
    if "match_type" in columns:
        select_cols.append("match_type")
    if "stage" in columns:
        select_cols.append("stage")

    matches = fetch_dataframe(conn, f"SELECT {', '.join(select_cols)} FROM matches")
    if "match_type" in matches.columns:
        matches["match_type"] = matches["match_type"].astype(str)
    else:
        matches["match_type"] = ""
    if "stage" in matches.columns:
        matches["stage"] = matches["stage"].astype(str)
    else:
        matches["stage"] = ""

    final_matches = matches[
        matches["match_type"].str.lower().eq("final")
        | matches["stage"].str.lower().eq("final")
    ].copy()

    if final_matches.empty:
        if "date" in matches.columns:
            matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
            final_matches = matches.sort_values("date").groupby("season").tail(1)
        else:
            final_matches = matches.sort_values("match_id").groupby("season").tail(1)

    titles = final_matches[["season", "winner"]].dropna()
    pivot = (
        titles.pivot_table(index="season", columns="winner", values="winner", aggfunc="count")
        .fillna(0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("IPL Dynasty Chart (Titles by Season)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Titles")

    save_plot(fig, output_dir / "ipl_dynasty_chart.png")


def plot_batsman_consistency(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot batsman consistency scatter."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT batsman, match_id, inning, batsman_runs, wide_runs, batting_team "
        "FROM deliveries",
    )
    deliveries["legal_ball"] = (deliveries["wide_runs"].fillna(0) == 0).astype(int)

    innings_runs = (
        deliveries.groupby(["batsman", "match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    agg = (
        innings_runs.groupby("batsman")
        .agg(avg_runs=("runs", "mean"), std_runs=("runs", "std"), innings=("runs", "count"))
        .reset_index()
    )
    agg = agg[agg["innings"] >= 50]
    agg["consistency_index"] = agg["avg_runs"] / agg["std_runs"].replace(0, pd.NA)

    team_runs = (
        deliveries.groupby(["batsman", "batting_team"])["batsman_runs"].sum().reset_index()
    )
    team_primary = team_runs.sort_values("batsman_runs", ascending=False).drop_duplicates("batsman")
    agg = agg.merge(team_primary[["batsman", "batting_team"]], on="batsman", how="left")

    team_colors = build_team_color_map(agg["batting_team"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(
        agg["avg_runs"],
        agg["consistency_index"],
        s=agg["innings"],
        c=agg["batting_team"].map(team_colors),
        alpha=0.75,
    )
    ax.set_title("Batsman Consistency: Avg Runs vs Consistency Index")
    ax.set_xlabel("Average Runs")
    ax.set_ylabel("Consistency Index")

    top_annotate = agg.nlargest(10, "consistency_index")
    for _, row in top_annotate.iterrows():
        ax.annotate(row["batsman"], (row["avg_runs"], row["consistency_index"]))

    save_plot(fig, output_dir / "batsman_consistency_scatter.png")


def plot_death_over_specialists(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot death over specialists bubble chart."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT bowler, over, total_runs, wide_runs, noball_runs, bye_runs, legbye_runs, penalty_runs, is_wicket "
        "FROM deliveries WHERE over BETWEEN 16 AND 20",
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
            wickets=("is_wicket", "sum"),
            dot_balls=("dot_ball", "sum"),
        )
        .reset_index()
    )
    agg["overs"] = agg["legal_balls"] / 6.0
    agg = agg[agg["overs"] >= 20]
    agg["economy"] = 6.0 * agg["runs_conceded"] / agg["legal_balls"].replace(0, pd.NA)
    agg["dot_ball_pct"] = 100.0 * agg["dot_balls"] / agg["legal_balls"].replace(0, pd.NA)

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        agg["economy"],
        agg["wickets"],
        s=agg["dot_ball_pct"] * 5,
        c=COLOR_PALETTE[4],
        alpha=0.7,
    )
    ax.set_title("Death Over Specialists")
    ax.set_xlabel("Economy (Overs 17-20)")
    ax.set_ylabel("Wickets")

    top_annotate = agg.sort_values(["economy", "wickets"], ascending=[True, False]).head(10)
    for _, row in top_annotate.iterrows():
        ax.annotate(row["bowler"], (row["economy"], row["wickets"]))

    save_plot(fig, output_dir / "death_over_specialists.png")


def main() -> None:
    """Generate all visualizations and save to results/."""
    apply_plot_style()
    project_root = get_project_root()
    db_path = project_root / "data" / "ipl.db"
    output_dir = project_root / "results"

    with sqlite3.connect(db_path) as conn:
        plot_top_run_scorers(conn, output_dir)
        plot_top_wicket_takers(conn, output_dir)
        plot_strike_rate_phase_heatmap(conn, output_dir)
        plot_economy_phase_heatmap(conn, output_dir)
        plot_toss_win_match_win(conn, output_dir)
        plot_venue_run_rates(conn, output_dir)
        plot_season_trends(conn, output_dir)
        plot_dynasty_chart(conn, output_dir)
        plot_batsman_consistency(conn, output_dir)
        plot_death_over_specialists(conn, output_dir)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
