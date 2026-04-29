"""Advanced batting analysis for IPL analytics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection."""
    if db_path is None:
        db_path = get_project_root() / "data" / "ipl.db"
    return sqlite3.connect(db_path)


def anchor_vs_aggressor_clustering(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Cluster batsmen into anchor vs aggressor profiles."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, batsman, batsman_runs, wide_runs, noball_runs "
            "FROM deliveries",
            conn,
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

    centroid_values = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(
        centroid_values, columns=["avg_runs", "strike_rate", "dot_ball_pct", "boundary_pct"]
    )
    print("\nCluster centroids (original scale):")
    print(centroid_df)

    aggressor_label = centroid_df["strike_rate"].idxmax()
    agg["cluster"] = np.where(labels == aggressor_label, "Aggressor", "Anchor")

    return agg[[
        "batsman",
        "cluster",
        "avg_runs",
        "strike_rate",
        "innings",
        "dot_ball_pct",
        "boundary_pct",
    ]].sort_values(["cluster", "strike_rate"], ascending=[True, False])


def career_trajectory(player_name: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return season-by-season runs, average, and strike rate for a player."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, season, batsman, batsman_runs, wide_runs, noball_runs, "
            "player_dismissed "
            "FROM deliveries",
            conn,
        )

    if player_name not in deliveries["batsman"].dropna().unique():
        print(f"Player not found: {player_name}")
        return pd.DataFrame()

    player = deliveries[deliveries["batsman"] == player_name].copy()
    player["legal_ball"] = (
        (player["wide_runs"].fillna(0) == 0) & (player["noball_runs"].fillna(0) == 0)
    ).astype(int)

    innings = (
        player.groupby(["season", "match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )

    dismissals = (
        deliveries[deliveries["player_dismissed"] == player_name]
        .groupby("season")["player_dismissed"]
        .count()
        .rename("outs")
    )

    season_stats = (
        innings.groupby("season")
        .agg(runs=("runs", "sum"), balls=("balls", "sum"), innings=("runs", "count"))
        .reset_index()
    )
    season_stats = season_stats.merge(dismissals, on="season", how="left")
    season_stats["outs"] = season_stats["outs"].fillna(0)
    season_stats["average"] = season_stats["runs"] / season_stats["outs"].replace(0, pd.NA)
    season_stats["strike_rate"] = 100.0 * season_stats["runs"] / season_stats["balls"].replace(0, pd.NA)

    return season_stats.sort_values("season")


def age_vs_performance(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Aggregate batting performance by career year (proxy for experience)."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, season, batsman, batsman_runs, wide_runs, noball_runs "
            "FROM deliveries",
            conn,
        )

    deliveries["legal_ball"] = (
        (deliveries["wide_runs"].fillna(0) == 0)
        & (deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)

    innings = (
        deliveries.groupby(["batsman", "season", "match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )

    season_stats = (
        innings.groupby(["batsman", "season"])
        .agg(runs=("runs", "sum"), balls=("balls", "sum"), innings=("runs", "count"))
        .reset_index()
        .sort_values(["batsman", "season"])
    )
    season_stats["career_year"] = season_stats.groupby("batsman").cumcount() + 1

    agg = (
        season_stats.groupby("career_year")
        .agg(
            avg_runs=("runs", "mean"),
            total_runs=("runs", "sum"),
            total_balls=("balls", "sum"),
            innings=("innings", "sum"),
        )
        .reset_index()
    )
    agg["strike_rate"] = 100.0 * agg["total_runs"] / agg["total_balls"].replace(0, pd.NA)

    return agg[["career_year", "avg_runs", "strike_rate", "innings"]].sort_values("career_year")


def opening_partnership_analysis(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return opening partnership averages and 50+ stand frequency."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, over, ball, batsman, non_striker, total_runs, is_wicket "
            "FROM deliveries",
            conn,
        )

    deliveries = deliveries.copy()
    deliveries["ball_index"] = (deliveries["over"] - 1) * 6 + deliveries["ball"]

    first_ball = (
        deliveries.sort_values(["match_id", "inning", "ball_index"])
        .groupby(["match_id", "inning"], as_index=False)
        .first()
    )
    first_ball["batter1"] = np.where(
        first_ball["batsman"] < first_ball["non_striker"],
        first_ball["batsman"],
        first_ball["non_striker"],
    )
    first_ball["batter2"] = np.where(
        first_ball["batsman"] < first_ball["non_striker"],
        first_ball["non_striker"],
        first_ball["batsman"],
    )

    first_wicket = (
        deliveries[deliveries["is_wicket"] == 1]
        .groupby(["match_id", "inning"])["ball_index"]
        .min()
        .rename("wicket_ball")
    )

    deliveries = deliveries.merge(first_wicket, on=["match_id", "inning"], how="left")
    max_ball = deliveries.groupby(["match_id", "inning"])["ball_index"].transform("max")
    deliveries["wicket_ball"] = deliveries["wicket_ball"].fillna(max_ball)

    partnership_runs = (
        deliveries[deliveries["ball_index"] <= deliveries["wicket_ball"]]
        .groupby(["match_id", "inning"])["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "partnership_runs"})
    )

    combined = first_ball.merge(partnership_runs, on=["match_id", "inning"], how="left")
    agg = (
        combined.groupby(["batter1", "batter2"])
        .agg(
            innings_together=("match_id", "count"),
            avg_partnership=("partnership_runs", "mean"),
            fifty_plus_pct=(
                "partnership_runs",
                lambda x: 100.0 * (x >= 50).sum() / x.count() if x.count() else 0,
            ),
        )
        .reset_index()
    )

    agg = agg[agg["innings_together"] >= 10]
    agg["fifty_plus_flag"] = agg["avg_partnership"] >= 50
    return agg.sort_values(["fifty_plus_pct", "avg_partnership"], ascending=False)


def early_wicket_impact(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return average final scores based on first wicket timing buckets."""
    with _get_connection(db_path) as conn:
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, over, total_runs, is_wicket "
            "FROM deliveries",
            conn,
        )

    first_wicket = (
        deliveries[deliveries["is_wicket"] == 1]
        .groupby(["match_id", "inning"])["over"]
        .min()
        .rename("first_wicket_over")
    )
    innings_scores = (
        deliveries.groupby(["match_id", "inning"])["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "final_score"})
    )
    merged = innings_scores.merge(first_wicket, on=["match_id", "inning"], how="left")
    merged["first_wicket_over"] = merged["first_wicket_over"].fillna(20)

    def bucket(over: float) -> str:
        if 1 <= over <= 6:
            return "early"
        if 7 <= over <= 15:
            return "middle"
        return "late"

    merged["wicket_bucket"] = merged["first_wicket_over"].apply(bucket)
    agg = (
        merged.groupby("wicket_bucket")
        .agg(avg_final_score=("final_score", "mean"), innings=("match_id", "count"))
        .reset_index()
    )
    return agg.sort_values("wicket_bucket")


def main() -> None:
    """Run advanced batting analysis functions and print samples."""
    clusters = anchor_vs_aggressor_clustering()
    print("\nAnchor vs Aggressor (sample)")
    print(clusters.head(10))

    trajectory = career_trajectory("Virat Kohli")
    print("\nCareer Trajectory (sample)")
    print(trajectory.head(10))

    experience = age_vs_performance()
    print("\nAge vs Performance (sample)")
    print(experience.head(10))

    openings = opening_partnership_analysis()
    print("\nOpening Partnership Analysis (sample)")
    print(openings.head(10))

    impact = early_wicket_impact()
    print("\nEarly Wicket Impact (sample)")
    print(impact)


if __name__ == "__main__":
    main()
