"""Advanced team strategy analysis for IPL analytics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection."""
    if db_path is None:
        db_path = get_project_root() / "data" / "ipl.db"
    return sqlite3.connect(db_path)


def _load_matches_deliveries(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and deliveries tables."""
    matches = pd.read_sql_query(
        "SELECT match_id, season, venue, team1, team2, toss_winner, toss_decision, winner, "
        "result_margin, date, neutral_venue "
        "FROM matches",
        conn,
    )
    deliveries = pd.read_sql_query(
        "SELECT match_id, inning, over, batting_team, total_runs, is_wicket "
        "FROM deliveries",
        conn,
    )
    return matches, deliveries


def _venue_avg_scores(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.Series:
    """Return average first-innings score by venue."""
    first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_score")
        .reset_index()
    )
    merged = matches.merge(first_innings, on="match_id", how="left")
    return merged.groupby("venue")["first_innings_score"].mean()


def win_probability_model(db_path: Optional[Path] = None):
    """Train a logistic regression win probability model and return key outputs."""
    with _get_connection(db_path) as conn:
        matches, deliveries = _load_matches_deliveries(conn)

    venue_avg = _venue_avg_scores(matches, deliveries)

    powerplay = (
        deliveries[deliveries["over"].between(1, 6)]
        .groupby(["match_id", "inning", "batting_team"])
        .agg(
            powerplay_score=("total_runs", "sum"),
            pp_wickets_lost=("is_wicket", "sum"),
        )
        .reset_index()
    )

    match_meta = matches[[
        "match_id",
        "venue",
        "toss_winner",
        "toss_decision",
        "winner",
    ]].copy()

    df = powerplay.merge(match_meta, on="match_id", how="left")
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

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    print(f"Win probability model accuracy: {accuracy:.3f}")
    print(f"Win probability model ROC-AUC: {roc_auc:.3f}")

    coef = pipeline.named_steps["logisticregression"].coef_[0]
    feature_importance = pd.DataFrame(
        {"feature": features, "coefficient": coef}
    ).sort_values("coefficient", ascending=False)

    predictions = model_data.loc[X_test.index, ["match_id", "batting_team", "win"]].copy()
    predictions["predicted_prob"] = probs
    predictions["predicted_class"] = preds

    return pipeline, feature_importance, predictions


def score_predictor(db_path: Optional[Path] = None):
    """Train a linear regression score predictor and return outputs."""
    with _get_connection(db_path) as conn:
        matches, deliveries = _load_matches_deliveries(conn)

    venue_avg = _venue_avg_scores(matches, deliveries)

    over10 = (
        deliveries[deliveries["over"] <= 10]
        .groupby(["match_id", "inning", "batting_team"])
        .agg(
            runs_at_over10=("total_runs", "sum"),
            wickets_at_over10=("is_wicket", "sum"),
        )
        .reset_index()
    )

    final_scores = (
        deliveries.groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "final_score"})
    )

    df = over10.merge(final_scores, on=["match_id", "inning", "batting_team"], how="left")
    df = df.merge(matches[["match_id", "venue"]], on="match_id", how="left")
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
    print(f"Score predictor R2: {r2:.3f}")

    results = model_data.loc[X_test.index, ["match_id", "batting_team", "venue", "final_score"]].copy()
    results["predicted_score"] = preds

    return model, results, r2


def rolling_form(team_name: str, window: int = 5, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return rolling win percentage by season for a team."""
    with _get_connection(db_path) as conn:
        matches = pd.read_sql_query(
            "SELECT match_id, season, date, team1, team2, winner "
            "FROM matches",
            conn,
        )

    team_matches = matches[(matches["team1"] == team_name) | (matches["team2"] == team_name)].copy()
    if team_matches.empty:
        return pd.DataFrame()

    if "date" in team_matches.columns:
        team_matches["date"] = pd.to_datetime(team_matches["date"], errors="coerce")
        team_matches = team_matches.sort_values(["season", "date", "match_id"])
    else:
        team_matches = team_matches.sort_values(["season", "match_id"])

    team_matches["win"] = (team_matches["winner"] == team_name).astype(int)
    team_matches["match_number"] = team_matches.groupby("season").cumcount() + 1
    team_matches["rolling_win_pct"] = (
        team_matches.groupby("season")["win"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        * 100.0
    )

    return team_matches[["season", "match_number", "rolling_win_pct"]]


def home_away_analysis(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return home/away/neutral win rates for each team."""
    with _get_connection(db_path) as conn:
        matches = pd.read_sql_query(
            "SELECT match_id, team1, team2, venue, neutral_venue, winner "
            "FROM matches",
            conn,
        )

    team_rows = pd.concat(
        [
            matches[["match_id", "team1", "venue", "neutral_venue", "winner"]].rename(
                columns={"team1": "team"}
            ),
            matches[["match_id", "team2", "venue", "neutral_venue", "winner"]].rename(
                columns={"team2": "team"}
            ),
        ],
        ignore_index=True,
    )

    home_venues = (
        team_rows.groupby(["team", "venue"]).size().reset_index(name="matches")
        .sort_values(["team", "matches"], ascending=[True, False])
        .drop_duplicates("team")
        .set_index("team")["venue"]
    )

    team_rows["home_venue"] = team_rows["team"].map(home_venues)
    team_rows["location"] = np.where(
        team_rows["neutral_venue"].fillna(0) == 1,
        "neutral",
        np.where(team_rows["venue"] == team_rows["home_venue"], "home", "away"),
    )
    team_rows["win"] = (team_rows["winner"] == team_rows["team"]).astype(int)

    agg = (
        team_rows.groupby(["team", "location"])
        .agg(matches=("match_id", "count"), wins=("win", "sum"))
        .reset_index()
    )
    agg["win_pct"] = 100.0 * agg["wins"] / agg["matches"].replace(0, pd.NA)
    return agg.sort_values(["team", "location"])


def chasing_by_target(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return chase success rate by target bucket."""
    with _get_connection(db_path) as conn:
        matches, deliveries = _load_matches_deliveries(conn)

    innings_totals = (
        deliveries.groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    targets = innings_totals[innings_totals["inning"] == 1][["match_id", "total_runs"]]
    targets = targets.rename(columns={"total_runs": "target"})
    targets["target"] = targets["target"] + 1

    chasing = innings_totals[innings_totals["inning"] == 2][["match_id", "batting_team"]]
    chasing = chasing.rename(columns={"batting_team": "chasing_team"})

    df = targets.merge(chasing, on="match_id", how="inner").merge(
        matches[["match_id", "winner"]], on="match_id", how="left"
    )
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

    df["target_bucket"] = df["target"].apply(bucket)
    agg = (
        df.groupby("target_bucket")
        .agg(matches=("match_id", "count"), chase_success_rate=("chase_win", "mean"))
        .reset_index()
    )
    agg["chase_success_rate"] = agg["chase_success_rate"] * 100.0
    return agg.sort_values("target_bucket")


def rivalry_breakdown(team_a: str, team_b: str, db_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Return rivalry breakdown tables for two teams."""
    with _get_connection(db_path) as conn:
        matches = pd.read_sql_query(
            "SELECT match_id, season, date, team1, team2, venue, toss_winner, toss_decision, "
            "winner, result_margin "
            "FROM matches",
            conn,
        )
        deliveries = pd.read_sql_query(
            "SELECT match_id, inning, batting_team, total_runs "
            "FROM deliveries",
            conn,
        )

    rivalry = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()

    if rivalry.empty:
        return {
            "record": pd.DataFrame(),
            "avg_scores": pd.DataFrame(),
            "toss_impact": pd.DataFrame(),
            "venue_breakdown": pd.DataFrame(),
            "timeline": pd.DataFrame(),
            "biggest_wins": pd.DataFrame(),
            "streak": pd.DataFrame(),
        }

    record = (
        rivalry.groupby("winner")
        .size()
        .reset_index(name="wins")
        .rename(columns={"winner": "team"})
    )

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

    toss_impact = rivalry.copy()
    toss_impact["toss_win"] = (toss_impact["toss_winner"] == toss_impact["winner"]).astype(int)
    toss_impact = (
        toss_impact.groupby("toss_winner")["toss_win"].mean().reset_index()
        .rename(columns={"toss_winner": "team", "toss_win": "win_rate_when_winning_toss"})
    )
    toss_impact["win_rate_when_winning_toss"] = toss_impact["win_rate_when_winning_toss"] * 100.0

    venue_breakdown = (
        rivalry.groupby(["venue", "winner"]).size().reset_index(name="wins")
    )

    timeline = rivalry.copy()
    if "date" in timeline.columns:
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
        timeline = timeline.sort_values(["date", "match_id"])
    else:
        timeline = timeline.sort_values("match_id")
    timeline = timeline.tail(10)

    biggest_wins = (
        rivalry.groupby("winner")["result_margin"]
        .max()
        .reset_index()
        .rename(columns={"winner": "team", "result_margin": "biggest_margin"})
    )

    streak_team = timeline["winner"].iloc[-1]
    streak_count = 0
    for winner in reversed(timeline["winner"].tolist()):
        if winner == streak_team:
            streak_count += 1
        else:
            break
    streak = pd.DataFrame({"team": [streak_team], "streak": [streak_count]})

    return {
        "record": record,
        "avg_scores": avg_scores,
        "toss_impact": toss_impact,
        "venue_breakdown": venue_breakdown,
        "timeline": timeline,
        "biggest_wins": biggest_wins,
        "streak": streak,
    }


def main() -> None:
    """Run advanced team analysis and print samples."""
    model, features, predictions = win_probability_model()
    print("\nWin Model Feature Importance (sample)")
    print(features)

    _, results, r2 = score_predictor()
    print("\nScore Predictor R2")
    print(r2)
    print(results.head(10))

    form = rolling_form("Mumbai Indians")
    print("\nRolling Form (sample)")
    print(form.head(10))

    home_away = home_away_analysis()
    print("\nHome/Away (sample)")
    print(home_away.head(10))

    chasing = chasing_by_target()
    print("\nChasing by Target (sample)")
    print(chasing)

    rivalry = rivalry_breakdown("Mumbai Indians", "Chennai Super Kings")
    print("\nRivalry Record (sample)")
    print(rivalry["record"].head(10))


if __name__ == "__main__":
    main()
