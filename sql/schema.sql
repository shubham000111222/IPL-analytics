-- Schema for IPL analytics database

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    season TEXT,
    date TEXT,
    team1 TEXT,
    team2 TEXT,
    venue TEXT,
    city TEXT,
    toss_winner TEXT,
    toss_decision TEXT,
    winner TEXT,
    result TEXT,
    result_margin INTEGER,
    player_of_match TEXT,
    umpire1 TEXT,
    umpire2 TEXT,
    neutral_venue INTEGER
);

CREATE TABLE IF NOT EXISTS deliveries (
    match_id INTEGER,
    inning INTEGER,
    over INTEGER,
    ball INTEGER,
    batting_team TEXT,
    bowling_team TEXT,
    batsman TEXT,
    non_striker TEXT,
    bowler TEXT,
    batsman_runs INTEGER,
    extra_runs INTEGER,
    total_runs INTEGER,
    wide_runs INTEGER,
    noball_runs INTEGER,
    bye_runs INTEGER,
    legbye_runs INTEGER,
    penalty_runs INTEGER,
    is_wicket INTEGER,
    dismissal_kind TEXT,
    player_dismissed TEXT,
    fielder TEXT,
    season TEXT,
    venue TEXT,
    match_winner TEXT
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match_id ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_batsman ON deliveries(batsman);
CREATE INDEX IF NOT EXISTS idx_deliveries_bowler ON deliveries(bowler);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
