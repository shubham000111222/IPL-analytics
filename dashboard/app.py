"""Streamlit dashboard for IPL analytics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


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


if __name__ == "__main__":
    main()
