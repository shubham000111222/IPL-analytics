"""Create publication-quality IPL analytics plots."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
            "font.family": "DejaVu Sans",
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


def plot_anchor_vs_aggressor_scatter(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot anchor vs aggressor clustering scatter."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, inning, batsman, batsman_runs, wide_runs, noball_runs "
        "FROM deliveries",
    )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)
    deliveries["dot_ball"] = (
        (deliveries["legal_ball"] == 1) & (deliveries["batsman_runs"] == 0)
    ).astype(int)
    deliveries["boundary_runs"] = np.where(
        deliveries["batsman_runs"].isin([4, 6]), deliveries["batsman_runs"], 0
    )

    innings = (
        deliveries.groupby(["batsman", "match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    innings_count = innings.groupby("batsman")["runs"].count().rename("innings")
    avg_runs = innings.groupby("batsman")["runs"].mean().rename("avg_runs")

    agg = (
        deliveries.groupby("batsman")
        .agg(
            total_runs=("batsman_runs", "sum"),
            total_balls=("legal_ball", "sum"),
            dot_balls=("dot_ball", "sum"),
            boundary_runs=("boundary_runs", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(innings_count, on="batsman", how="left")
    agg = agg.merge(avg_runs, on="batsman", how="left")
    agg = agg[agg["innings"] >= 50].copy()

    agg["strike_rate"] = 100.0 * agg["total_runs"] / agg["total_balls"].replace(0, pd.NA)
    agg["dot_ball_pct"] = 100.0 * agg["dot_balls"] / agg["total_balls"].replace(0, pd.NA)
    agg["boundary_pct"] = 100.0 * agg["boundary_runs"] / agg["total_runs"].replace(0, pd.NA)

    features = agg[["avg_runs", "strike_rate", "dot_ball_pct", "boundary_pct"]].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(
        centroids, columns=["avg_runs", "strike_rate", "dot_ball_pct", "boundary_pct"]
    )
    print("\nCluster centroids (original scale):")
    print(centroid_df)

    aggressor_label = centroid_df["strike_rate"].idxmax()
    agg["cluster"] = np.where(labels == aggressor_label, "Aggressor", "Anchor")

    color_map = {"Anchor": "#1f77b4", "Aggressor": "#d62728"}
    sizes = 30 + (agg["innings"] - agg["innings"].min()) * 2

    fig, ax = plt.subplots(figsize=(12, 7))
    for cluster, group in agg.groupby("cluster"):
        ax.scatter(
            group["avg_runs"],
            group["strike_rate"],
            s=sizes.loc[group.index],
            color=color_map.get(cluster, "#333333"),
            alpha=0.75,
            label=cluster,
        )

    median_avg = agg["avg_runs"].median()
    median_sr = agg["strike_rate"].median()
    ax.axvline(median_avg, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(median_sr, color="#777777", linestyle="--", linewidth=1)

    top_annotate = agg.nlargest(15, "total_runs")
    for _, row in top_annotate.iterrows():
        ax.annotate(row["batsman"], (row["avg_runs"], row["strike_rate"]))

    ax.set_title("IPL Batting Profile Clusters: Anchors vs Aggressors")
    ax.set_xlabel("Career Average")
    ax.set_ylabel("Career Strike Rate")
    ax.legend()

    save_plot(fig, output_dir / "batting_clusters.png")


def plot_bowler_batsman_matchup_heatmap(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot matchup heatmap for top batsmen vs top bowlers."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT batsman, bowler, batsman_runs, wide_runs, noball_runs, dismissal_kind, is_wicket "
        "FROM deliveries",
    )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)

    top_batsmen = deliveries.groupby("batsman")["batsman_runs"].sum().nlargest(15).index
    wicket_mask = (
        (deliveries["is_wicket"] == 1)
        & (~deliveries["dismissal_kind"].fillna("").str.lower().isin(
            ["run out", "retired hurt", "retired out", "obstructing the field"]
        ))
    )
    top_bowlers = deliveries[wicket_mask].groupby("bowler")["is_wicket"].sum().nlargest(15).index

    filtered = deliveries[
        deliveries["batsman"].isin(top_batsmen)
        & deliveries["bowler"].isin(top_bowlers)
    ].copy()

    agg = (
        filtered.groupby(["batsman", "bowler"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    agg["economy"] = 6.0 * agg["runs"] / agg["balls"].replace(0, pd.NA)
    pivot = agg.pivot(index="batsman", columns="bowler", values="economy")

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color="#bdbdbd")
    annot = pivot.apply(lambda col: col.map(lambda x: "" if pd.isna(x) else f"{x:.1f}"))
    sns.heatmap(pivot, cmap=cmap, annot=annot, fmt="", ax=ax, cbar_kws={"label": "Economy"})
    ax.set_title("Head-to-Head: Top Batsmen vs Top Bowlers (Economy Rate)")
    ax.set_xlabel("Bowler")
    ax.set_ylabel("Batsman")

    save_plot(fig, output_dir / "matchup_matrix.png")


def plot_chasing_success_by_target(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot chasing success rate by target bucket."""
    innings_totals = fetch_dataframe(
        conn,
        "SELECT match_id, inning, batting_team, SUM(total_runs) AS runs "
        "FROM deliveries GROUP BY match_id, inning, batting_team",
    )
    matches = fetch_dataframe(conn, "SELECT match_id, winner FROM matches")

    targets = innings_totals[innings_totals["inning"] == 1][["match_id", "runs"]].copy()
    targets["target"] = targets["runs"] + 1
    chasing = innings_totals[innings_totals["inning"] == 2][["match_id", "batting_team"]].copy()
    chasing = chasing.rename(columns={"batting_team": "chasing_team"})

    df = targets.merge(chasing, on="match_id", how="inner").merge(matches, on="match_id", how="left")
    df["chase_win"] = (df["winner"] == df["chasing_team"]).astype(int)

    def bucket(target: float) -> str:
        if 140 <= target <= 150:
            return "140-150"
        if 151 <= target <= 160:
            return "151-160"
        if 161 <= target <= 170:
            return "161-170"
        if 171 <= target <= 180:
            return "171-180"
        if 181 <= target <= 190:
            return "181-190"
        if 191 <= target <= 200:
            return "191-200"
        if target >= 201:
            return "200+"
        return "<140"

    df["bucket"] = df["target"].apply(bucket)
    agg = (
        df.groupby("bucket")
        .agg(matches=("match_id", "count"), success=("chase_win", "mean"))
        .reset_index()
    )
    agg["success"] = agg["success"] * 100.0

    order = ["<140", "140-150", "151-160", "161-170", "171-180", "181-190", "191-200", "200+"]
    agg["bucket"] = pd.Categorical(agg["bucket"], categories=order, ordered=True)
    agg = agg.sort_values("bucket")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("RdYlGn_r", n_colors=len(agg))
    bars = ax.bar(agg["bucket"].astype(str), agg["success"], color=colors)
    ax.axhline(50, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("How Target Size Kills Chase Success Rate")
    ax.set_xlabel("Target Range")
    ax.set_ylabel("Chase Success (%)")

    for bar, matches in zip(bars, agg["matches"]):
        ax.annotate(
            f"{int(matches)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    save_plot(fig, output_dir / "chasing_success.png")


def plot_toss_decision_trend(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot the trend of choosing to field first after winning the toss."""
    matches = fetch_dataframe(conn, "SELECT season, toss_decision FROM matches")
    matches["field_first"] = matches["toss_decision"].str.lower().eq("field").astype(int)
    trend = (
        matches.groupby("season")
        .agg(field_first_pct=("field_first", "mean"), matches=("field_first", "count"))
        .reset_index()
    )
    trend["field_first_pct"] = trend["field_first_pct"] * 100.0
    trend = trend.sort_values("season")

    trend["diff"] = trend["field_first_pct"].diff()
    inflection = trend.loc[trend["diff"].idxmax()] if trend["diff"].notna().any() else None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trend["season"], trend["field_first_pct"], marker="o", color=COLOR_PALETTE[2])
    ax.fill_between(
        trend["season"],
        50,
        trend["field_first_pct"],
        where=trend["field_first_pct"] >= 50,
        color=COLOR_PALETTE[2],
        alpha=0.15,
    )
    if inflection is not None:
        ax.annotate(
            f"Inflection: {int(inflection['season'])}",
            (inflection["season"], inflection["field_first_pct"]),
            textcoords="offset points",
            xytext=(10, 10),
        )

    ax.set_title("The Fielding-First Revolution in IPL (2008-2024)")
    ax.set_xlabel("Season")
    ax.set_ylabel("% Toss Winners Choosing to Field")

    save_plot(fig, output_dir / "toss_trend.png")


def plot_run_rate_pattern(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot average runs per over for winning vs losing teams."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, over, batting_team, total_runs FROM deliveries",
    )
    matches = fetch_dataframe(conn, "SELECT match_id, winner FROM matches")

    over_runs = (
        deliveries.groupby(["match_id", "over", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )
    over_runs = over_runs.merge(matches, on="match_id", how="left")
    over_runs["outcome"] = np.where(
        over_runs["batting_team"] == over_runs["winner"], "win", "loss"
    )

    agg = (
        over_runs.groupby(["over", "outcome"])["total_runs"]
        .mean()
        .reset_index()
        .rename(columns={"total_runs": "avg_runs"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for outcome, color in zip(["win", "loss"], [COLOR_PALETTE[1], COLOR_PALETTE[5]]):
        subset = agg[agg["outcome"] == outcome]
        ax.plot(subset["over"], subset["avg_runs"], marker="o", label=outcome.title(), color=color)

    ax.axvspan(1, 6.5, color="#cfe8ff", alpha=0.2)
    ax.axvspan(15.5, 20.5, color="#ffe6cc", alpha=0.2)

    ax.set_title("Where Matches Are Won and Lost: Run Rate by Over")
    ax.set_xlabel("Over")
    ax.set_ylabel("Avg Runs per Over")
    ax.legend()

    save_plot(fig, output_dir / "run_rate_pattern.png")


def plot_win_probability_feature_importance(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot logistic regression feature importance for win probability."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, inning, over, batting_team, total_runs, is_wicket, venue, "
        "toss_winner, match_winner "
        "FROM deliveries",
    )

    powerplay = (
        deliveries[deliveries["over"].between(1, 6)]
        .groupby(["match_id", "inning", "batting_team", "venue", "toss_winner", "match_winner"])
        .agg(
            powerplay_score=("total_runs", "sum"),
            pp_wickets_lost=("is_wicket", "sum"),
        )
        .reset_index()
    )

    first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_score")
        .reset_index()
    )
    venue_avg = (
        deliveries[["match_id", "venue"]]
        .drop_duplicates("match_id")
        .merge(first_innings, on="match_id", how="left")
        .groupby("venue")["first_innings_score"]
        .mean()
    )

    powerplay["batting_first"] = (powerplay["inning"] == 1).astype(int)
    powerplay["toss_won"] = (powerplay["batting_team"] == powerplay["toss_winner"]).astype(int)
    powerplay["venue_avg_score"] = powerplay["venue"].map(venue_avg)
    powerplay["win"] = (powerplay["batting_team"] == powerplay["match_winner"]).astype(int)

    features = ["powerplay_score", "pp_wickets_lost", "toss_won", "batting_first", "venue_avg_score"]
    model_data = powerplay.dropna(subset=features + ["win"]).copy()
    model_data[features] = model_data[features].fillna(model_data[features].median())

    X = model_data[features]
    y = model_data["win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)

    coef = pipeline.named_steps["logisticregression"].coef_[0]
    importance = pd.DataFrame({"feature": features, "coefficient": coef})
    importance = importance.sort_values("coefficient")

    colors = ["#d62728" if val < 0 else "#2ca02c" for val in importance["coefficient"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance["feature"], importance["coefficient"], color=colors)
    ax.set_title("What Actually Wins IPL Matches? (Logistic Regression)")
    ax.set_xlabel("Coefficient")

    save_plot(fig, output_dir / "win_factors.png")


def plot_score_predictor(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot actual vs predicted final scores from a linear regression model."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, inning, over, batting_team, total_runs, is_wicket, venue "
        "FROM deliveries",
    )

    over10 = (
        deliveries[deliveries["over"] <= 10]
        .groupby(["match_id", "inning", "batting_team", "venue"])
        .agg(
            runs_at_over10=("total_runs", "sum"),
            wickets_at_over10=("is_wicket", "sum"),
        )
        .reset_index()
    )

    final_scores = (
        deliveries.groupby(["match_id", "inning", "batting_team", "venue"])["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "final_score"})
    )

    df = over10.merge(final_scores, on=["match_id", "inning", "batting_team", "venue"], how="left")
    venue_avg = (
        deliveries[deliveries["inning"] == 1]
        .groupby(["match_id", "venue"])["total_runs"]
        .sum()
        .reset_index()
        .groupby("venue")["total_runs"]
        .mean()
    )
    df["venue_avg"] = df["venue"].map(venue_avg)

    features = ["runs_at_over10", "wickets_at_over10", "venue_avg"]
    model_data = df.dropna(subset=features + ["final_score"]).copy()
    model_data[features] = model_data[features].fillna(model_data[features].median())

    X = model_data[features]
    y = model_data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    plot_df = model_data.loc[X_test.index, ["final_score", "venue"]].copy()
    plot_df["predicted_score"] = preds

    top_venues = plot_df["venue"].value_counts().head(8).index
    plot_df["venue_group"] = np.where(plot_df["venue"].isin(top_venues), plot_df["venue"], "Other")

    fig, ax = plt.subplots(figsize=(10, 7))
    for venue, group in plot_df.groupby("venue_group"):
        ax.scatter(group["predicted_score"], group["final_score"], alpha=0.7, label=venue)

    min_val = min(plot_df["predicted_score"].min(), plot_df["final_score"].min())
    max_val = max(plot_df["predicted_score"].max(), plot_df["final_score"].max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#333333", linestyle="--")

    ax.set_title("Predicting Final Score from 10-Over Position")
    ax.set_xlabel("Predicted Final Score")
    ax.set_ylabel("Actual Final Score")
    ax.text(0.02, 0.95, f"R2 = {r2:.3f}", transform=ax.transAxes)
    ax.legend(title="Venue", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot(fig, output_dir / "score_predictor.png")


def plot_player_radar(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot a radar chart comparing two batters."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT match_id, inning, over, batsman, batsman_runs, wide_runs, noball_runs, "
        "player_dismissed "
        "FROM deliveries",
    )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)
    deliveries["boundary_runs"] = np.where(
        deliveries["batsman_runs"].isin([4, 6]), deliveries["batsman_runs"], 0
    )

    innings = (
        deliveries.groupby(["batsman", "match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )

    outs = (
        deliveries[deliveries["player_dismissed"].notna()]
        .groupby("player_dismissed")["player_dismissed"]
        .count()
        .rename("outs")
    )

    agg = (
        deliveries.groupby("batsman")
        .agg(
            total_runs=("batsman_runs", "sum"),
            total_balls=("legal_ball", "sum"),
            boundary_runs=("boundary_runs", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(outs, left_on="batsman", right_index=True, how="left")
    agg["outs"] = agg["outs"].fillna(0)
    agg["average"] = agg["total_runs"] / agg["outs"].replace(0, pd.NA)
    agg["strike_rate"] = 100.0 * agg["total_runs"] / agg["total_balls"].replace(0, pd.NA)
    agg["boundary_pct"] = 100.0 * agg["boundary_runs"] / agg["total_runs"].replace(0, pd.NA)

    consistency = (
        innings.groupby("batsman")
        .agg(avg_runs=("runs", "mean"), std_runs=("runs", "std"), innings=("runs", "count"))
        .reset_index()
    )
    consistency["consistency"] = consistency["avg_runs"] / consistency["std_runs"].replace(0, pd.NA)

    phase = deliveries.copy()
    phase["phase"] = pd.cut(
        phase["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    )
    phase_agg = (
        phase.groupby(["batsman", "phase"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    phase_agg["sr"] = 100.0 * phase_agg["runs"] / phase_agg["balls"].replace(0, pd.NA)
    phase_pivot = phase_agg.pivot(index="batsman", columns="phase", values="sr")

    metrics = (
        agg.merge(consistency[["batsman", "consistency", "innings"]], on="batsman", how="left")
        .merge(phase_pivot, on="batsman", how="left")
    )

    metrics = metrics[metrics["innings"] >= 30].copy()

    metric_cols = {
        "Avg": "average",
        "SR": "strike_rate",
        "Boundary%": "boundary_pct",
        "Consistency": "consistency",
        "Powerplay SR": "Powerplay",
        "Death SR": "Death",
    }

    ranges = {}
    for label, col in metric_cols.items():
        values = metrics[col].dropna()
        ranges[label] = (values.min(), values.max())

    def normalize(value: float, label: str) -> float:
        min_val, max_val = ranges[label]
        if pd.isna(value) or max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val) * 100.0

    default_players = ["Virat Kohli", "Rohit Sharma"]
    available = metrics["batsman"].tolist()
    if all(player in available for player in default_players):
        player_a, player_b = default_players
    else:
        top_two = metrics.sort_values("total_runs", ascending=False).head(2)["batsman"].tolist()
        player_a, player_b = top_two[0], top_two[1]

    def player_values(player: str) -> list[float]:
        row = metrics[metrics["batsman"] == player].iloc[0]
        return [normalize(row[col], label) for label, col in metric_cols.items()]

    labels = list(metric_cols.keys())
    values_a = player_values(player_a)
    values_b = player_values(player_b)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values_a, color=COLOR_PALETTE[0], linewidth=2, label=player_a)
    ax.fill(angles, values_a, color=COLOR_PALETTE[0], alpha=0.25)
    ax.plot(angles, values_b, color=COLOR_PALETTE[5], linewidth=2, label=player_b)
    ax.fill(angles, values_b, color=COLOR_PALETTE[5], alpha=0.2)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.set_title("Player Radar Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_plot(fig, output_dir / "player_radar.png")


def plot_wicket_type_distribution_heatmap(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot wicket type distribution heatmap for top bowlers."""
    deliveries = fetch_dataframe(
        conn,
        "SELECT bowler, dismissal_kind, is_wicket FROM deliveries",
    )

    wickets = deliveries[deliveries["is_wicket"] == 1].copy()
    wickets["dismissal_kind"] = wickets["dismissal_kind"].fillna("").str.lower()

    def classify(kind: str) -> str:
        if "bowled" in kind:
            return "bowled"
        if "caught" in kind:
            return "caught"
        if "lbw" in kind:
            return "lbw"
        if "stumped" in kind:
            return "stumped"
        if "run out" in kind:
            return "run out"
        return "other"

    wickets["wicket_type"] = wickets["dismissal_kind"].apply(classify)

    top_bowlers = wickets.groupby("bowler")["is_wicket"].sum().nlargest(20).index
    wickets = wickets[wickets["bowler"].isin(top_bowlers)]

    totals = wickets.groupby("bowler")["is_wicket"].sum().rename("total_wickets")
    dist = (
        wickets[wickets["wicket_type"].isin(["bowled", "caught", "lbw", "stumped", "run out"])]
        .groupby(["bowler", "wicket_type"])
        .size()
        .reset_index(name="count")
    )
    dist = dist.merge(totals, on="bowler", how="left")
    dist["pct"] = 100.0 * dist["count"] / dist["total_wickets"].replace(0, pd.NA)

    pivot = dist.pivot(index="bowler", columns="wicket_type", values="pct").fillna(0)
    pivot = pivot[["bowled", "caught", "lbw", "stumped", "run out"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Wicket Type Distribution by Bowler")
    ax.set_xlabel("Wicket Type")
    ax.set_ylabel("Bowler")

    save_plot(fig, output_dir / "wicket_types.png")


def plot_head_to_head_win_matrix(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Plot head-to-head win matrix for all teams."""
    matches = fetch_dataframe(
        conn,
        "SELECT match_id, team1, team2, winner FROM matches",
    )

    teams = sorted(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()))
    win_matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)
    match_matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)

    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                win_matrix.loc[team_a, team_b] = np.nan
                match_matrix.loc[team_a, team_b] = np.nan
                continue
            subset = matches[
                ((matches["team1"] == team_a) & (matches["team2"] == team_b))
                | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
            ]
            if subset.empty:
                win_matrix.loc[team_a, team_b] = np.nan
                match_matrix.loc[team_a, team_b] = np.nan
                continue
            wins = (subset["winner"] == team_a).sum()
            matches_played = subset.shape[0]
            win_matrix.loc[team_a, team_b] = 100.0 * wins / matches_played
            match_matrix.loc[team_a, team_b] = matches_played

    annot = win_matrix.copy().astype(object)
    for row in teams:
        for col in teams:
            if pd.isna(win_matrix.loc[row, col]):
                annot.loc[row, col] = ""
            else:
                pct = win_matrix.loc[row, col]
                matches_played = int(match_matrix.loc[row, col])
                annot.loc[row, col] = f"{pct:.0f}%\n({matches_played})"

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color="#bdbdbd")
    sns.heatmap(
        win_matrix,
        cmap=cmap,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="#ffffff",
        cbar_kws={"label": "Win %"},
        ax=ax,
    )
    ax.set_title("IPL Head-to-Head Win Matrix (All Time)")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Team")

    save_plot(fig, output_dir / "h2h_matrix.png")


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
        plot_anchor_vs_aggressor_scatter(conn, output_dir)
        plot_bowler_batsman_matchup_heatmap(conn, output_dir)
        plot_chasing_success_by_target(conn, output_dir)
        plot_toss_decision_trend(conn, output_dir)
        plot_run_rate_pattern(conn, output_dir)
        plot_win_probability_feature_importance(conn, output_dir)
        plot_score_predictor(conn, output_dir)
        plot_player_radar(conn, output_dir)
        plot_wicket_type_distribution_heatmap(conn, output_dir)
        plot_head_to_head_win_matrix(conn, output_dir)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
