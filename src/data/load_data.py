"""Load, clean, and persist IPL data into SQLite."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


TEAM_NAME_MAP: Dict[str, str] = {
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "royal challengers bengaluru": "Royal Challengers Bangalore",
    "royal challengers bangalore": "Royal Challengers Bangalore",
    "chennai super kings": "Chennai Super Kings",
    "mumbai indians": "Mumbai Indians",
    "kolkata knight riders": "Kolkata Knight Riders",
    "rajasthan royals": "Rajasthan Royals",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "gujarat lions": "Gujarat Lions",
    "gujarat titans": "Gujarat Titans",
    "lucknow super giants": "Lucknow Super Giants",
    "lucknow supergiants": "Lucknow Super Giants",
    "pune warriors india": "Pune Warriors",
    "pune warriors": "Pune Warriors",
    "kochi tuskers kerala": "Kochi Tuskers Kerala",
    "deccan chargers": "Deccan Chargers",
}

TEAM_COLUMNS_MATCHES = ["team1", "team2", "winner", "toss_winner"]
TEAM_COLUMNS_DELIVERIES = ["batting_team", "bowling_team", "fielding_team"]
PLAYER_COLUMNS = ["batsman", "non_striker", "bowler", "player_dismissed", "fielder"]


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case."""
    rename_map = {col: re.sub(r"\s+", "_", col.strip().lower()) for col in df.columns}
    return df.rename(columns=rename_map)


def standardize_team_names(series: pd.Series) -> pd.Series:
    """Standardize team names using a predefined mapping."""
    def normalize_team(value: object) -> object:
        if pd.isna(value):
            return value
        cleaned = re.sub(r"\s+", " ", str(value).strip())
        return TEAM_NAME_MAP.get(cleaned.lower(), cleaned)

    return series.apply(normalize_team)


def standardize_player_names(series: pd.Series) -> pd.Series:
    """Standardize player names by trimming whitespace and normalizing casing."""
    def normalize_name(value: object) -> object:
        if pd.isna(value):
            return value
        name = re.sub(r"\s+", " ", str(value).strip())
        if name.islower():
            return name.title()
        return name

    return series.apply(normalize_name)


def detect_and_rename_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and rename match columns to a consistent schema."""
    df = normalize_column_names(df)

    rename_map = {}
    if "id" in df.columns and "match_id" not in df.columns:
        rename_map["id"] = "match_id"
    if "start_date" in df.columns and "date" not in df.columns:
        rename_map["start_date"] = "date"
    if "season_year" in df.columns and "season" not in df.columns:
        rename_map["season_year"] = "season"

    df = df.rename(columns=rename_map)

    if "match_id" not in df.columns:
        raise KeyError("matches.csv must contain a match identifier column")

    if "neutral_venue" not in df.columns:
        df["neutral_venue"] = 0

    return df


def detect_and_rename_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and rename delivery columns to a consistent schema."""
    df = normalize_column_names(df)

    rename_map = {}
    if "id" in df.columns and "match_id" not in df.columns:
        rename_map["id"] = "match_id"
    if "batter" in df.columns and "batsman" not in df.columns:
        rename_map["batter"] = "batsman"
    if "ball_number" in df.columns and "ball" not in df.columns:
        rename_map["ball_number"] = "ball"

    df = df.rename(columns=rename_map)

    if "match_id" not in df.columns:
        raise KeyError("deliveries.csv must contain a match identifier column")

    return df


def ensure_columns(df: pd.DataFrame, columns: Dict[str, object]) -> pd.DataFrame:
    """Ensure columns exist with default values."""
    for col, default in columns.items():
        if col not in df.columns:
            df[col] = default
    return df


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric with missing values filled as 0."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the matches dataset."""
    df = detect_and_rename_matches(df)

    for col in TEAM_COLUMNS_MATCHES:
        if col in df.columns:
            df[col] = standardize_team_names(df[col])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.year

    if "winner" in df.columns:
        df["winner"] = df["winner"].fillna("No Result")

    return df


def prepare_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the deliveries dataset."""
    df = detect_and_rename_deliveries(df)

    for col in TEAM_COLUMNS_DELIVERIES:
        if col in df.columns:
            df[col] = standardize_team_names(df[col])

    for col in PLAYER_COLUMNS:
        if col in df.columns:
            df[col] = standardize_player_names(df[col])

    df = ensure_columns(
        df,
        {
            "wide_runs": 0,
            "noball_runs": 0,
            "bye_runs": 0,
            "legbye_runs": 0,
            "penalty_runs": 0,
            "dismissal_kind": np.nan,
            "player_dismissed": np.nan,
        },
    )

    df = coerce_numeric(
        df,
        [
            "inning",
            "over",
            "ball",
            "batsman_runs",
            "extra_runs",
            "total_runs",
            "wide_runs",
            "noball_runs",
            "bye_runs",
            "legbye_runs",
            "penalty_runs",
        ],
    )

    if "over" in df.columns and df["over"].min() == 0:
        df["over"] = df["over"] + 1
    if "ball" in df.columns and df["ball"].min() == 0:
        df["ball"] = df["ball"] + 1

    if "is_wicket" not in df.columns:
        df["is_wicket"] = df["player_dismissed"].notna().astype(int)

    return df


def merge_deliveries_matches(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Merge match metadata into deliveries."""
    for col in ["season", "venue", "winner", "team1", "team2", "toss_winner", "toss_decision"]:
        if col not in matches.columns:
            matches[col] = np.nan

    match_meta = matches[[
        "match_id",
        "season",
        "venue",
        "winner",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
    ]].copy()
    match_meta = match_meta.rename(columns={"winner": "match_winner"})
    merged = deliveries.merge(match_meta, on="match_id", how="left")
    merged["match_winner"] = merged["match_winner"].fillna("No Result")
    return merged


def create_sqlite_db(matches: pd.DataFrame, deliveries: pd.DataFrame, db_path: Path) -> None:
    """Create SQLite database with matches and deliveries tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        matches.to_sql("matches", conn, if_exists="replace", index=False)
        deliveries.to_sql("deliveries", conn, if_exists="replace", index=False)


def data_quality_report(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Print a data quality report for matches and deliveries."""
    print("\n=== Data Quality Report ===")
    print(f"Matches shape: {matches.shape}")
    print(f"Deliveries shape: {deliveries.shape}")

    if "date" in matches.columns:
        date_min = matches["date"].min()
        date_max = matches["date"].max()
        print(f"Date range: {date_min} to {date_max}")

    team_cols = [col for col in TEAM_COLUMNS_MATCHES if col in matches.columns]
    team_values = pd.concat([matches[col] for col in team_cols], axis=0)
    team_values = team_values.dropna().unique()
    print(f"Unique teams: {len(team_values)}")

    player_cols = [col for col in PLAYER_COLUMNS if col in deliveries.columns]
    player_values = pd.concat([deliveries[col] for col in player_cols], axis=0)
    player_values = player_values.dropna().unique()
    print(f"Unique players: {len(player_values)}")

    print("\nNull counts (matches):")
    print(matches.isna().sum().sort_values(ascending=False).head(10))

    print("\nNull counts (deliveries):")
    print(deliveries.isna().sum().sort_values(ascending=False).head(10))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with safe defaults."""
    return pd.read_csv(path)


def main() -> None:
    """Run the full data load and persistence pipeline."""
    project_root = get_project_root()
    data_dir = project_root / "data" / "raw"
    matches_path = data_dir / "matches.csv"
    deliveries_path = data_dir / "deliveries.csv"

    if not matches_path.exists() or not deliveries_path.exists():
        raise FileNotFoundError("Missing matches.csv or deliveries.csv in data/raw")

    matches = load_csv(matches_path)
    deliveries = load_csv(deliveries_path)

    matches = prepare_matches(matches)
    deliveries = prepare_deliveries(deliveries)

    deliveries = merge_deliveries_matches(deliveries, matches)

    db_path = project_root / "data" / "ipl.db"
    create_sqlite_db(matches, deliveries, db_path)

    data_quality_report(matches, deliveries)
    print(f"\nSQLite database created at: {db_path}")


if __name__ == "__main__":
    main()
