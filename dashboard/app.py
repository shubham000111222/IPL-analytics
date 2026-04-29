"""Streamlit dashboard for IPL analytics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


@st.cache_data
def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@st.cache_data
def get_db_path() -> Path:
    """Return the path to the SQLite database."""
    return get_project_root() / "data" / "ipl.db"


def database_has_tables(db_path: Path) -> bool:
    """Return True if the SQLite DB exists and has matches/deliveries tables."""
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {row[0] for row in rows}
    return {"matches", "deliveries"}.issubset(tables)


def ensure_database() -> None:
    """Create the SQLite database from CSVs if missing."""
    db_path = get_db_path()
    if database_has_tables(db_path):
        return

    root = get_project_root()
    data_dir = root / "data" / "raw"
    matches_path = data_dir / "matches.csv"
    deliveries_path = data_dir / "deliveries.csv"

    if not matches_path.exists() or not deliveries_path.exists():
        st.error("Missing data/raw CSVs. Add matches.csv and deliveries.csv to data/raw.")
        st.stop()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.data.load_data import (
        create_sqlite_db,
        merge_deliveries_matches,
        prepare_deliveries,
        prepare_matches,
    )

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    matches = prepare_matches(matches)
    deliveries = prepare_deliveries(deliveries)
    deliveries = merge_deliveries_matches(deliveries, matches)
    create_sqlite_db(matches, deliveries, db_path)


@st.cache_data
def load_table(table: str) -> pd.DataFrame:
    """Load a table from SQLite into a DataFrame."""
    ensure_database()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


@st.cache_data
def load_sql_queries() -> Dict[str, str]:
    """Load SQL queries with names for the SQL explorer."""
    root = get_project_root()
    sql_files = [
        root / "sql" / "batting_analysis.sql",
        root / "sql" / "bowling_analysis.sql",
        root / "sql" / "team_strategy.sql",
    ]

    queries: Dict[str, str] = {}
    for file_path in sql_files:
        text = file_path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in text.split("-- name:") if block.strip()]
        for block in blocks:
            lines = block.splitlines()
            name = lines[0].strip()
            sql = "\n".join(lines[1:]).strip()
            if sql.endswith(";"):
                sql = sql[:-1]
            queries[name] = sql
    return queries


def filter_by_season(matches: pd.DataFrame, deliveries: pd.DataFrame, season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter matches and deliveries by season if selected."""
    if season == "All":
        return matches, deliveries

    matches_filtered = matches[matches["season"].astype(str) == season]
    deliveries_filtered = deliveries[deliveries["season"].astype(str) == season]
    return matches_filtered, deliveries_filtered


def render_overview(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render the Overview dashboard page."""
    st.header("Overview")

    seasons = ["All"] + sorted(matches["season"].dropna().astype(str).unique())
    season = st.selectbox("Season", seasons)

    matches_filtered, deliveries_filtered = filter_by_season(matches, deliveries, season)

    total_matches = matches_filtered["match_id"].nunique()
    total_runs = deliveries_filtered["total_runs"].sum()
    total_sixes = (deliveries_filtered["batsman_runs"] == 6).sum()

    innings_scores = (
        deliveries_filtered.groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )
    highest_score = innings_scores["total_runs"].max() if not innings_scores.empty else 0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Matches", int(total_matches))
    kpi_cols[1].metric("Total Runs", int(total_runs))
    kpi_cols[2].metric("Total Sixes", int(total_sixes))
    kpi_cols[3].metric("Highest Team Score", int(highest_score))

    season_trends = (
        deliveries.groupby(["season", "match_id"])
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            boundaries=("batsman_runs", lambda x: (x == 4).sum() * 4 + (x == 6).sum() * 6),
        )
        .reset_index()
    )
    season_summary = (
        season_trends.groupby("season")
        .agg(
            avg_runs=("runs", "mean"),
            avg_wickets=("wickets", "mean"),
            runs_sum=("runs", "sum"),
            boundary_runs_sum=("boundaries", "sum"),
        )
        .reset_index()
    )
    season_summary["boundary_pct"] = (
        100.0
        * season_summary["boundary_runs_sum"]
        / season_summary["runs_sum"].replace(0, pd.NA)
    )

    line_fig = px.line(
        season_summary,
        x="season",
        y=["avg_runs", "avg_wickets", "boundary_pct"],
        markers=True,
        title="Season Trends",
    )
    st.plotly_chart(line_fig, use_container_width=True)

    titles = compute_titles_table(matches)
    st.subheader("IPL Titles")
    st.dataframe(titles, use_container_width=True)


def compute_titles_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute IPL titles by season."""
    matches = matches.copy()
    if "match_type" in matches.columns:
        final_matches = matches[matches["match_type"].str.lower() == "final"]
    elif "stage" in matches.columns:
        final_matches = matches[matches["stage"].str.lower() == "final"]
    else:
        if "date" in matches.columns:
            matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
            final_matches = matches.sort_values("date").groupby("season").tail(1)
        else:
            final_matches = matches.sort_values("match_id").groupby("season").tail(1)

    titles = (
        final_matches[["season", "winner"]]
        .dropna()
        .rename(columns={"winner": "champion"})
        .sort_values("season")
    )
    return titles


def render_batting_analysis(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render the Batting Analysis dashboard page."""
    st.header("Batting Analysis")

    batsmen = sorted(deliveries["batsman"].dropna().unique())
    selected = st.selectbox("Select Batsman", batsmen)

    player_deliveries = deliveries[deliveries["batsman"] == selected].copy()
    innings = (
        player_deliveries.groupby(["match_id", "inning"])
        .agg(runs=("batsman_runs", "sum"), balls=("wide_runs", lambda x: (x == 0).sum()))
        .reset_index()
    )

    matches_played = innings["match_id"].nunique()
    total_runs = innings["runs"].sum()
    outs = (
        deliveries[deliveries["player_dismissed"] == selected]
        .groupby("match_id")["player_dismissed"]
        .count()
        .sum()
    )
    average = total_runs / outs if outs else 0
    balls_faced = innings["balls"].sum()
    strike_rate = 100.0 * total_runs / balls_faced if balls_faced else 0

    fifties = (innings["runs"].between(50, 99)).sum()
    hundreds = (innings["runs"] >= 100).sum()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Matches", int(matches_played))
    kpi_cols[1].metric("Runs", int(total_runs))
    kpi_cols[2].metric("Avg", round(average, 2))
    kpi_cols[3].metric("SR", round(strike_rate, 2))
    kpi_cols[4].metric("50s / 100s", f"{int(fifties)} / {int(hundreds)}")

    st.subheader("Run Distribution")
    hist_fig = px.histogram(innings, x="runs", nbins=25, title="Innings Runs Distribution")
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Strike Rate by Phase")
    player_deliveries["phase"] = pd.cut(
        player_deliveries["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    )
    phase_agg = (
        player_deliveries.groupby("phase")
        .agg(runs=("batsman_runs", "sum"), balls=("wide_runs", lambda x: (x == 0).sum()))
        .reset_index()
    )
    phase_agg["strike_rate"] = 100.0 * phase_agg["runs"] / phase_agg["balls"].replace(0, pd.NA)
    phase_fig = px.bar(phase_agg, x="phase", y="strike_rate", title="Strike Rate by Phase")
    st.plotly_chart(phase_fig, use_container_width=True)

    st.subheader("Season-by-Season Runs")
    season_runs = (
        player_deliveries.groupby("season")["batsman_runs"].sum().reset_index()
    )
    season_fig = px.line(season_runs, x="season", y="batsman_runs", markers=True)
    st.plotly_chart(season_fig, use_container_width=True)


def render_bowling_analysis(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render the Bowling Analysis dashboard page."""
    st.header("Bowling Analysis")

    bowlers = sorted(deliveries["bowler"].dropna().unique())
    selected = st.selectbox("Select Bowler", bowlers)

    player_deliveries = deliveries[deliveries["bowler"] == selected].copy()
    player_deliveries["legal_ball"] = (
        (player_deliveries["wide_runs"].fillna(0) == 0)
        & (player_deliveries["noball_runs"].fillna(0) == 0)
    ).astype(int)
    player_deliveries["runs_conceded"] = (
        player_deliveries["total_runs"].fillna(0)
        - player_deliveries["bye_runs"].fillna(0)
        - player_deliveries["legbye_runs"].fillna(0)
        - player_deliveries["penalty_runs"].fillna(0)
    )

    matches_played = player_deliveries["match_id"].nunique()
    wickets = player_deliveries["is_wicket"].sum()
    overs = player_deliveries["legal_ball"].sum() / 6.0
    economy = player_deliveries["runs_conceded"].sum() / overs if overs else 0
    bowling_avg = player_deliveries["runs_conceded"].sum() / wickets if wickets else 0
    strike_rate = player_deliveries["legal_ball"].sum() / wickets if wickets else 0

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Matches", int(matches_played))
    kpi_cols[1].metric("Wickets", int(wickets))
    kpi_cols[2].metric("Economy", round(economy, 2))
    kpi_cols[3].metric("Average", round(bowling_avg, 2))
    kpi_cols[4].metric("Strike Rate", round(strike_rate, 2))

    st.subheader("Wicket Type Distribution")
    wicket_types = (
        player_deliveries[player_deliveries["is_wicket"] == 1]
        .groupby("dismissal_kind")
        .size()
        .reset_index(name="count")
    )
    pie_fig = px.pie(wicket_types, names="dismissal_kind", values="count")
    st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("Economy by Phase")
    player_deliveries["phase"] = pd.cut(
        player_deliveries["over"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    )
    phase_agg = (
        player_deliveries.groupby("phase")
        .agg(runs=("runs_conceded", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )
    phase_agg["economy"] = 6.0 * phase_agg["runs"] / phase_agg["balls"].replace(0, pd.NA)
    phase_fig = px.bar(phase_agg, x="phase", y="economy", title="Economy by Phase")
    st.plotly_chart(phase_fig, use_container_width=True)

    st.subheader("Season-by-Season Wickets")
    season_wickets = player_deliveries.groupby("season")["is_wicket"].sum().reset_index()
    season_fig = px.line(season_wickets, x="season", y="is_wicket", markers=True)
    st.plotly_chart(season_fig, use_container_width=True)


def render_team_strategy(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render the Team Strategy dashboard page."""
    st.header("Team Strategy")

    teams = sorted(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()))
    selected = st.selectbox("Select Team", teams)

    team_matches = matches[(matches["team1"] == selected) | (matches["team2"] == selected)].copy()
    toss_team = team_matches[team_matches["toss_winner"] == selected].copy()
    toss_team["win"] = (toss_team["winner"] == selected).astype(int)
    toss_agg = (
        toss_team.groupby("toss_decision")["win"].mean().reset_index().assign(win_rate=lambda df: df["win"] * 100.0)
    )

    st.subheader("Win Rate by Toss Decision")
    toss_fig = px.bar(toss_agg, x="toss_decision", y="win_rate", title="Win Rate by Toss Decision")
    st.plotly_chart(toss_fig, use_container_width=True)

    st.subheader("Home vs Away Record")
    team_rows = pd.concat(
        [
            matches[["match_id", "team1", "winner", "city"]].rename(columns={"team1": "team"}),
            matches[["match_id", "team2", "winner", "city"]].rename(columns={"team2": "team"}),
        ],
        ignore_index=True,
    )
    team_rows = team_rows[team_rows["team"] == selected]

    def is_home(city: str) -> bool:
        return city in TEAM_HOME_CITIES.get(selected, [])

    team_rows["home_away"] = team_rows["city"].apply(lambda city: "home" if is_home(str(city)) else "away")
    team_rows["win"] = (team_rows["winner"] == selected).astype(int)
    home_away = (
        team_rows.groupby("home_away")["win"].mean().reset_index().assign(win_rate=lambda df: df["win"] * 100.0)
    )
    home_fig = px.bar(home_away, x="home_away", y="win_rate", title="Home vs Away Win Rate")
    st.plotly_chart(home_fig, use_container_width=True)

    st.subheader("Head-to-Head Win Matrix")
    opponents = teams
    matrix = pd.DataFrame(index=opponents, columns=opponents, dtype=float)
    for opponent in opponents:
        subset = matches[(matches["team1"].isin([selected, opponent])) & (matches["team2"].isin([selected, opponent]))]
        if subset.empty:
            matrix.loc[selected, opponent] = 0
        else:
            matrix.loc[selected, opponent] = 100.0 * (subset["winner"] == selected).mean()
    heatmap_fig = px.imshow(matrix.loc[[selected]], aspect="auto", title="Win Rate vs Opponents")
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("Best Venue")
    venue_perf = (
        team_matches.groupby("venue")
        .agg(matches=("match_id", "nunique"), wins=("winner", lambda x: (x == selected).sum()))
        .reset_index()
    )
    venue_perf = venue_perf[venue_perf["matches"] >= 5]
    venue_perf["win_rate"] = 100.0 * venue_perf["wins"] / venue_perf["matches"].replace(0, pd.NA)
    best_venue = venue_perf.sort_values("win_rate", ascending=False).head(1)
    st.dataframe(best_venue, use_container_width=True)


WICKET_EXCLUSIONS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}


@st.cache_data
def compute_batting_summary() -> pd.DataFrame:
    """Compute batting metrics for all players."""
    deliveries = load_table("deliveries").copy()
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
    innings_summary = (
        innings.groupby("batsman")
        .agg(
            innings=("runs", "count"),
            avg_runs=("runs", "mean"),
            std_runs=("runs", "std"),
            fifties=("runs", lambda x: (x.between(50, 99)).sum()),
            hundreds=("runs", lambda x: (x >= 100).sum()),
        )
        .reset_index()
    )

    totals = (
        deliveries.groupby("batsman")
        .agg(
            matches=("match_id", "nunique"),
            runs=("batsman_runs", "sum"),
            balls=("legal_ball", "sum"),
            boundary_runs=("boundary_runs", "sum"),
        )
        .reset_index()
    )

    outs = (
        deliveries[deliveries["player_dismissed"].notna()]
        .groupby("player_dismissed")["player_dismissed"]
        .count()
        .rename("outs")
        .reset_index()
        .rename(columns={"player_dismissed": "batsman"})
    )

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

    summary = (
        totals.merge(innings_summary, on="batsman", how="left")
        .merge(outs, on="batsman", how="left")
        .merge(phase_pivot, on="batsman", how="left")
    )
    summary["outs"] = summary["outs"].fillna(0)
    summary["average"] = summary["runs"] / summary["outs"].replace(0, pd.NA)
    summary["strike_rate"] = 100.0 * summary["runs"] / summary["balls"].replace(0, pd.NA)
    summary["boundary_pct"] = 100.0 * summary["boundary_runs"] / summary["runs"].replace(0, pd.NA)
    summary["consistency"] = summary["avg_runs"] / summary["std_runs"].replace(0, pd.NA)

    return summary


@st.cache_data
def compute_bowling_summary() -> pd.DataFrame:
    """Compute bowling metrics for all players."""
    deliveries = load_table("deliveries").copy()
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
    deliveries["dot_ball"] = (
        (deliveries["total_runs"].fillna(0) == 0) & (deliveries["legal_ball"] == 1)
    ).astype(int)
    deliveries["bowler_wicket"] = (
        (deliveries["is_wicket"] == 1)
        & (~deliveries["dismissal_kind"].fillna("").str.lower().isin(WICKET_EXCLUSIONS))
    ).astype(int)

    agg = (
        deliveries.groupby("bowler")
        .agg(
            matches=("match_id", "nunique"),
            legal_balls=("legal_ball", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            wickets=("bowler_wicket", "sum"),
            dot_balls=("dot_ball", "sum"),
        )
        .reset_index()
    )
    agg["overs"] = agg["legal_balls"] / 6.0
    agg["economy"] = agg["runs_conceded"] / agg["overs"].replace(0, pd.NA)
    agg["average"] = agg["runs_conceded"] / agg["wickets"].replace(0, pd.NA)
    agg["strike_rate"] = agg["legal_balls"] / agg["wickets"].replace(0, pd.NA)
    agg["dot_ball_pct"] = 100.0 * agg["dot_balls"] / agg["legal_balls"].replace(0, pd.NA)

    return agg


@st.cache_data
def compute_player_trajectory(player_name: str) -> pd.DataFrame:
    """Return a player's season-by-season trajectory."""
    deliveries = load_table("deliveries").copy()
    if player_name not in deliveries["batsman"].dropna().unique():
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

    outs = (
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
    season_stats = season_stats.merge(outs, on="season", how="left")
    season_stats["outs"] = season_stats["outs"].fillna(0)
    season_stats["average"] = season_stats["runs"] / season_stats["outs"].replace(0, pd.NA)
    season_stats["strike_rate"] = 100.0 * season_stats["runs"] / season_stats["balls"].replace(0, pd.NA)

    return season_stats.sort_values("season")


def compute_player_head_to_head(
    deliveries: pd.DataFrame, player_a: str, player_b: str
) -> pd.DataFrame:
    """Return head-to-head batting record for two players."""
    runs_a = (
        deliveries[deliveries["batsman"] == player_a]
        .groupby("match_id")["batsman_runs"]
        .sum()
    )
    runs_b = (
        deliveries[deliveries["batsman"] == player_b]
        .groupby("match_id")["batsman_runs"]
        .sum()
    )

    common = runs_a.index.intersection(runs_b.index)
    if common.empty:
        return pd.DataFrame()

    compare = pd.DataFrame({"runs_a": runs_a.loc[common], "runs_b": runs_b.loc[common]})
    a_better = (compare["runs_a"] > compare["runs_b"]).sum()
    b_better = (compare["runs_b"] > compare["runs_a"]).sum()
    ties = (compare["runs_a"] == compare["runs_b"]).sum()

    return pd.DataFrame(
        {
            "matches": [len(compare)],
            "player_a_better": [a_better],
            "player_b_better": [b_better],
            "ties": [ties],
        }
    )


@st.cache_data
def train_win_probability_model() -> Dict[str, object]:
    """Train win probability model and return training artifacts."""
    deliveries = load_table("deliveries")
    matches = load_table("matches")

    powerplay = (
        deliveries[deliveries["over"].between(1, 6)]
        .groupby(["match_id", "inning", "batting_team"])
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
        matches[["match_id", "venue"]]
        .merge(first_innings, on="match_id", how="left")
        .groupby("venue")["first_innings_score"]
        .mean()
    )

    team_pp_avg = powerplay.groupby("batting_team").agg(
        powerplay_score=("powerplay_score", "mean"),
        pp_wickets_lost=("pp_wickets_lost", "mean"),
    )

    df = powerplay.merge(
        matches[["match_id", "venue", "toss_winner", "toss_decision", "winner"]],
        on="match_id",
        how="left",
    )
    df["batting_first"] = (df["inning"] == 1).astype(int)
    df["toss_won"] = (df["batting_team"] == df["toss_winner"]).astype(int)
    df["venue_avg_score"] = df["venue"].map(venue_avg)
    df["win"] = (df["batting_team"] == df["winner"]).astype(int)

    features = ["powerplay_score", "pp_wickets_lost", "toss_won", "batting_first", "venue_avg_score"]
    model_data = df.dropna(subset=features + ["win"]).copy()
    model_data[features] = model_data[features].fillna(model_data[features].median())

    X = model_data[features]
    y = model_data["win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    coef = pipeline.named_steps["logisticregression"].coef_[0]
    importance = pd.DataFrame({"feature": features, "coefficient": coef})
    importance["abs"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs", ascending=False)

    venue_counts = matches["venue"].value_counts()

    return {
        "model": pipeline,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "importance": importance,
        "team_pp_avg": team_pp_avg,
        "venue_avg": venue_avg,
        "venue_counts": venue_counts,
    }


def render_player_comparison(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render player comparison dashboard page."""
    st.header("Player Comparison")

    players = sorted(deliveries["batsman"].dropna().unique())
    if not players:
        st.warning("No player data available.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("Player A", players, index=0)
    with col_b:
        player_b = st.selectbox("Player B", players, index=1 if len(players) > 1 else 0)

    batting_summary = compute_batting_summary()
    bowling_summary = compute_bowling_summary()

    row_a = batting_summary[batting_summary["batsman"] == player_a]
    row_b = batting_summary[batting_summary["batsman"] == player_b]

    if row_a.empty or row_b.empty:
        st.warning("Player not found.")
        return

    def batting_card(row: pd.Series, container: st.container) -> None:
        container.subheader("Batting")
        cols = container.columns(3)
        cols[0].metric("Avg", round(row.get("average", 0), 2))
        cols[1].metric("SR", round(row.get("strike_rate", 0), 2))
        cols[2].metric("Innings", int(row.get("innings", 0)))
        cols = container.columns(3)
        cols[0].metric("50s", int(row.get("fifties", 0)))
        cols[1].metric("100s", int(row.get("hundreds", 0)))
        cols[2].metric("Boundary%", round(row.get("boundary_pct", 0), 1))
        container.metric("Consistency", round(row.get("consistency", 0), 2))

    def bowling_card(row: pd.Series, container: st.container) -> None:
        container.subheader("Bowling")
        cols = container.columns(3)
        cols[0].metric("Wickets", int(row.get("wickets", 0)))
        cols[1].metric("Economy", round(row.get("economy", 0), 2))
        cols[2].metric("Average", round(row.get("average", 0), 2))
        cols = container.columns(2)
        cols[0].metric("Strike Rate", round(row.get("strike_rate", 0), 2))
        cols[1].metric("Dot Ball%", round(row.get("dot_ball_pct", 0), 1))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(player_a)
        batting_card(row_a.iloc[0], col_a)
        bowl_row_a = bowling_summary[bowling_summary["bowler"] == player_a]
        if not bowl_row_a.empty:
            bowling_card(bowl_row_a.iloc[0], col_a)
        else:
            st.caption("No bowling data available.")

    with col_b:
        st.subheader(player_b)
        batting_card(row_b.iloc[0], col_b)
        bowl_row_b = bowling_summary[bowling_summary["bowler"] == player_b]
        if not bowl_row_b.empty:
            bowling_card(bowl_row_b.iloc[0], col_b)
        else:
            st.caption("No bowling data available.")

    st.subheader("Radar Comparison")
    radar_metrics = {
        "Avg": "average",
        "SR": "strike_rate",
        "Boundary%": "boundary_pct",
        "Consistency": "consistency",
        "Powerplay SR": "Powerplay",
        "Death SR": "Death",
    }
    radar_source = batting_summary[batting_summary["innings"] >= 30].copy()

    def normalize(value: float, label: str) -> float:
        values = radar_source[radar_metrics[label]].dropna()
        min_val, max_val = values.min(), values.max()
        if values.empty or pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val) * 100.0

    def player_values(row: pd.Series) -> List[float]:
        return [normalize(row[radar_metrics[label]], label) for label in radar_metrics]

    values_a = player_values(row_a.iloc[0])
    values_b = player_values(row_b.iloc[0])
    labels = list(radar_metrics.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values_a, color="#1f77b4", linewidth=2, label=player_a)
    ax.fill(angles, values_a, color="#1f77b4", alpha=0.2)
    ax.plot(angles, values_b, color="#d62728", linewidth=2, label=player_b)
    ax.fill(angles, values_b, color="#d62728", alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)

    st.subheader("Head-to-Head (Same Matches)")
    head_to_head = compute_player_head_to_head(deliveries, player_a, player_b)
    if head_to_head.empty:
        st.info("No overlapping matches found for these players.")
    else:
        st.dataframe(head_to_head, use_container_width=True)

    st.subheader("Career Trajectory")
    trajectory_a = compute_player_trajectory(player_a)
    trajectory_b = compute_player_trajectory(player_b)
    if trajectory_a.empty or trajectory_b.empty:
        st.warning("Player not found.")
    else:
        trajectory_a = trajectory_a.assign(player=player_a)
        trajectory_b = trajectory_b.assign(player=player_b)
        combined = pd.concat([trajectory_a, trajectory_b], ignore_index=True)
        line_fig = px.line(
            combined,
            x="season",
            y="runs",
            color="player",
            markers=True,
            title="Season-by-Season Runs",
        )
        st.plotly_chart(line_fig, use_container_width=True)


def render_match_simulator(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render match simulator dashboard page."""
    st.header("Match Simulator")
    teams = sorted(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()))
    venues = sorted(matches["venue"].dropna().unique())

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        team_a = st.selectbox("Team A", teams)
    with col_b:
        team_b = st.selectbox("Team B", [t for t in teams if t != team_a])
    with col_c:
        venue = st.selectbox("Venue", venues)

    toss_winner = st.radio("Toss Winner", [team_a, team_b], horizontal=True)
    toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

    if st.button("Simulate"):
        artifacts = train_win_probability_model()
        model = artifacts["model"]
        accuracy = artifacts["accuracy"]
        importance = artifacts["importance"]
        team_pp_avg = artifacts["team_pp_avg"]
        venue_avg = artifacts["venue_avg"]
        venue_counts = artifacts["venue_counts"]

        if toss_decision == "bat":
            batting_first_team = toss_winner
        else:
            batting_first_team = team_a if toss_winner == team_b else team_b

        team_stats = team_pp_avg.loc[team_a] if team_a in team_pp_avg.index else None
        if team_stats is None:
            st.error("Insufficient powerplay data for the selected team.")
            return

        features = {
            "powerplay_score": team_stats["powerplay_score"],
            "pp_wickets_lost": team_stats["pp_wickets_lost"],
            "toss_won": 1 if toss_winner == team_a else 0,
            "batting_first": 1 if batting_first_team == team_a else 0,
            "venue_avg_score": venue_avg.get(venue, venue_avg.mean()),
        }
        input_df = pd.DataFrame([features])
        prob = float(model.predict_proba(input_df)[0][1])

        st.subheader("Win Probability")
        st.metric("Model Test Accuracy", f"{accuracy:.2%}")
        st.progress(prob)
        st.write(f"{team_a} win probability: {prob:.1%}")
        st.caption(f"Based on {int(venue_counts.get(venue, 0))} historical matches at this venue.")

        st.subheader("Top Influencing Factors")
        top_factors = importance.head(3).copy()
        top_factors["direction"] = np.where(top_factors["coefficient"] >= 0, "Helps", "Hurts")
        st.dataframe(top_factors[["feature", "direction", "coefficient"]], use_container_width=True)

        st.info("Based on historical data patterns, not a real predictor.")


def render_rivalry_deep_dive(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Render rivalry deep dive dashboard page."""
    st.header("Rivalry Deep Dive")

    teams = sorted(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()))
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A", teams, key="rivalry_team_a")
    with col_b:
        team_b = st.selectbox("Team B", [t for t in teams if t != team_a], key="rivalry_team_b")

    rivalry = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()

    if rivalry.empty:
        st.warning("No matches found for this rivalry.")
        return

    st.subheader("All-Time Record")
    wins_a = (rivalry["winner"] == team_a).sum()
    wins_b = (rivalry["winner"] == team_b).sum()
    no_result = rivalry[~rivalry["winner"].isin([team_a, team_b])].shape[0]
    cols = st.columns(3)
    cols[0].metric(team_a, wins_a)
    cols[1].metric(team_b, wins_b)
    cols[2].metric("No Result", no_result)

    innings = (
        deliveries[deliveries["match_id"].isin(rivalry["match_id"])]
        .groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )
    avg_scores = (
        innings[innings["inning"] == 1]
        .groupby("batting_team")["total_runs"]
        .mean()
        .reset_index()
        .rename(columns={"batting_team": "team", "total_runs": "avg_first_innings_score"})
    )
    st.subheader("Average First Innings Score")
    st.dataframe(avg_scores, use_container_width=True)

    toss_impact = rivalry.copy()
    toss_impact["toss_win"] = (toss_impact["toss_winner"] == toss_impact["winner"]).astype(int)
    toss_impact = (
        toss_impact.groupby("toss_winner")["toss_win"].mean().reset_index()
        .rename(columns={"toss_winner": "team", "toss_win": "win_rate_when_winning_toss"})
    )
    toss_impact["win_rate_when_winning_toss"] = toss_impact["win_rate_when_winning_toss"] * 100.0
    st.subheader("Toss Win Impact")
    st.dataframe(toss_impact, use_container_width=True)

    venue_breakdown = rivalry.groupby(["venue", "winner"]).size().reset_index(name="wins")
    st.subheader("Venue Breakdown")
    st.dataframe(venue_breakdown, use_container_width=True)

    timeline = rivalry.copy()
    if "date" in timeline.columns:
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
        timeline = timeline.sort_values(["date", "match_id"])
        timeline["order"] = timeline["date"]
    else:
        timeline = timeline.sort_values("match_id")
        timeline["order"] = timeline["match_id"]
    timeline = timeline.tail(10)

    st.subheader("Last 10 Meetings")
    base_line = go.Scatter(
        x=timeline["order"],
        y=[1] * len(timeline),
        mode="lines",
        line=dict(color="#cccccc"),
        showlegend=False,
    )
    fig = go.Figure(data=[base_line])
    for team, color in zip([team_a, team_b], ["#1f77b4", "#d62728"]):
        subset = timeline[timeline["winner"] == team]
        fig.add_trace(
            go.Scatter(
                x=subset["order"],
                y=[1] * len(subset),
                mode="markers",
                marker=dict(size=10, color=color),
                name=team,
                text=subset["season"],
            )
        )
    fig.update_yaxes(visible=False)
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Biggest Wins")
    biggest = (
        rivalry.groupby("winner")["result_margin"]
        .max()
        .reset_index()
        .rename(columns={"winner": "team", "result_margin": "biggest_margin"})
    )
    st.dataframe(biggest, use_container_width=True)

    streak_team = timeline["winner"].iloc[-1]
    streak_count = 0
    for winner in reversed(timeline["winner"].tolist()):
        if winner == streak_team:
            streak_count += 1
        else:
            break
    st.subheader("Current Winning Streak")
    st.metric("Streak", f"{streak_team}: {streak_count} wins")


def render_sql_explorer() -> None:
    """Render the SQL Explorer page."""
    st.header("SQL Explorer")
    queries = load_sql_queries()
    selected = st.selectbox("Select Query", sorted(queries.keys()))
    st.code(queries[selected], language="sql")

    if st.button("Run Query"):
        db_path = get_db_path()
        with sqlite3.connect(db_path) as conn:
            result = pd.read_sql_query(queries[selected], conn)
        st.dataframe(result, use_container_width=True)
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv",
        )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="IPL Analytics", layout="wide")
    st.title("IPL Analytics Dashboard")

    matches = load_table("matches")
    deliveries = load_table("deliveries")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Batting Analysis",
            "Bowling Analysis",
            "Team Strategy",
            "SQL Explorer",
            "Player Comparison",
            "Match Simulator",
            "Rivalry Deep Dive",
        ],
    )

    if page == "Overview":
        render_overview(matches, deliveries)
    elif page == "Batting Analysis":
        render_batting_analysis(matches, deliveries)
    elif page == "Bowling Analysis":
        render_bowling_analysis(matches, deliveries)
    elif page == "Team Strategy":
        render_team_strategy(matches, deliveries)
    elif page == "SQL Explorer":
        render_sql_explorer()
    elif page == "Player Comparison":
        render_player_comparison(matches, deliveries)
    elif page == "Match Simulator":
        render_match_simulator(matches, deliveries)
    elif page == "Rivalry Deep Dive":
        render_rivalry_deep_dive(matches, deliveries)


if __name__ == "__main__":
    main()
