-- name: Toss Impact on Win Rate
-- Business question: How does batting first vs chasing affect win rate by team and venue type?
WITH base AS (
    SELECT
        match_id,
        season,
        venue,
        team1,
        team2,
        toss_winner,
        toss_decision,
        winner,
        CASE WHEN COALESCE(neutral_venue, 0) = 1 THEN 'neutral' ELSE 'home' END AS venue_type,
        CASE
            WHEN lower(toss_decision) = 'bat' THEN toss_winner
            WHEN lower(toss_decision) = 'field' THEN CASE WHEN toss_winner = team1 THEN team2 ELSE team1 END
            ELSE NULL
        END AS batting_first
    FROM matches
), team_roles AS (
    SELECT match_id, season, venue_type, batting_first AS team, 'bat_first' AS role, winner, team1, team2
    FROM base
    UNION ALL
    SELECT match_id, season, venue_type,
           CASE WHEN batting_first = team1 THEN team2 ELSE team1 END AS team,
           'chase' AS role,
           winner,
           team1,
           team2
    FROM base
)
SELECT
    team,
    venue_type,
    role,
    COUNT(*) AS matches,
    SUM(CASE WHEN team = winner THEN 1 ELSE 0 END) AS wins,
    100.0 * SUM(CASE WHEN team = winner THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS win_rate
FROM team_roles
GROUP BY team, venue_type, role
ORDER BY win_rate DESC;

-- name: Powerplay Score vs Match Outcome
-- Business question: Do higher powerplay scores translate into match wins across seasons?
WITH powerplay AS (
    SELECT
        match_id,
        season,
        batting_team,
        inning,
        SUM(total_runs) AS powerplay_runs,
        MAX(match_winner) AS match_winner
    FROM deliveries
    WHERE over BETWEEN 1 AND 6
    GROUP BY match_id, season, batting_team, inning
), outcome AS (
    SELECT
        match_id,
        season,
        batting_team,
        powerplay_runs,
        CASE WHEN batting_team = match_winner THEN 'win' ELSE 'loss' END AS outcome
    FROM powerplay
)
SELECT
    season,
    outcome,
    AVG(powerplay_runs) AS avg_powerplay_runs,
    COUNT(*) AS innings
FROM outcome
GROUP BY season, outcome
ORDER BY season, outcome;

-- name: Venue Analysis
-- Business question: Which venues favor high first-innings scores and successful chases?
WITH innings_totals AS (
    SELECT match_id, inning, batting_team, SUM(total_runs) AS runs
    FROM deliveries
    GROUP BY match_id, inning, batting_team
), first_innings AS (
    SELECT match_id, runs AS first_innings_runs
    FROM innings_totals
    WHERE inning = 1
), winning_score AS (
    SELECT i.match_id, i.runs AS winning_score
    FROM innings_totals i
    JOIN matches m ON m.match_id = i.match_id
    WHERE i.batting_team = m.winner
), chase_success AS (
    SELECT m.match_id,
           CASE WHEN m.winner = i2.batting_team THEN 1 ELSE 0 END AS chase_win
    FROM matches m
    LEFT JOIN innings_totals i2 ON i2.match_id = m.match_id AND i2.inning = 2
)
SELECT
    m.venue,
    COUNT(*) AS matches,
    AVG(f.first_innings_runs) AS avg_first_innings_score,
    AVG(w.winning_score) AS avg_winning_score,
    100.0 * SUM(c.chase_win) / NULLIF(COUNT(*), 0) AS chasing_success_rate
FROM matches m
LEFT JOIN first_innings f ON f.match_id = m.match_id
LEFT JOIN winning_score w ON w.match_id = m.match_id
LEFT JOIN chase_success c ON c.match_id = m.match_id
GROUP BY m.venue
HAVING COUNT(*) >= 10
ORDER BY avg_first_innings_score DESC;

-- name: Partnership Analysis
-- Business question: Which batting partnerships have produced the most runs together?
WITH pairs AS (
    SELECT
        match_id,
        inning,
        CASE WHEN batsman < non_striker THEN batsman ELSE non_striker END AS batter1,
        CASE WHEN batsman < non_striker THEN non_striker ELSE batsman END AS batter2,
        batsman_runs,
        CASE WHEN COALESCE(wide_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball
    FROM deliveries
), pair_agg AS (
    SELECT
        batter1,
        batter2,
        SUM(batsman_runs) AS runs,
        SUM(legal_ball) AS balls
    FROM pairs
    GROUP BY batter1, batter2
)
SELECT
    batter1,
    batter2,
    runs,
    balls,
    100.0 * runs / NULLIF(balls, 0) AS strike_rate
FROM pair_agg
ORDER BY runs DESC
LIMIT 20;

-- name: Win Probability by Run Rate
-- Business question: How does the required run rate impact chase win probability?
WITH innings_totals AS (
    SELECT match_id, inning, SUM(total_runs) AS runs
    FROM deliveries
    GROUP BY match_id, inning
), targets AS (
    SELECT match_id, runs + 1 AS target
    FROM innings_totals
    WHERE inning = 1
), over_runs AS (
    SELECT match_id, inning, over, SUM(total_runs) AS runs_in_over
    FROM deliveries
    WHERE inning = 2
    GROUP BY match_id, inning, over
), over_cum AS (
    SELECT
        o.match_id,
        o.over,
        SUM(o.runs_in_over) OVER (PARTITION BY o.match_id ORDER BY o.over) AS runs_scored,
        (20 - o.over) AS overs_left
    FROM over_runs o
), rr AS (
    SELECT
        c.match_id,
        c.over,
        t.target,
        c.runs_scored,
        (t.target - c.runs_scored) * 1.0 / NULLIF(c.overs_left, 0) AS required_run_rate
    FROM over_cum c
    JOIN targets t ON t.match_id = c.match_id
), rr_bracket AS (
    SELECT
        match_id,
        over,
        required_run_rate,
        CASE
            WHEN required_run_rate < 6 THEN '<6'
            WHEN required_run_rate >= 6 AND required_run_rate < 8 THEN '6-8'
            WHEN required_run_rate >= 8 AND required_run_rate < 10 THEN '8-10'
            WHEN required_run_rate >= 10 AND required_run_rate < 12 THEN '10-12'
            WHEN required_run_rate >= 12 AND required_run_rate < 14 THEN '12-14'
            ELSE '14+'
        END AS rr_bucket
    FROM rr
), chasing_team AS (
    SELECT match_id, MAX(batting_team) AS chasing_team
    FROM deliveries
    WHERE inning = 2
    GROUP BY match_id
)
SELECT
    r.rr_bucket,
    COUNT(*) AS overs_observed,
    100.0 * SUM(CASE WHEN m.winner = c.chasing_team THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS win_rate
FROM rr_bracket r
JOIN chasing_team c ON c.match_id = r.match_id
JOIN matches m ON m.match_id = r.match_id
GROUP BY r.rr_bucket
ORDER BY r.rr_bucket;

-- name: Season-on-Season Team Performance
-- Business question: How do team win rates and efficiency metrics change year over year?
WITH team_matches AS (
    SELECT match_id, season, team1 AS team, CASE WHEN team1 = winner THEN 1 ELSE 0 END AS win
    FROM matches
    UNION ALL
    SELECT match_id, season, team2 AS team, CASE WHEN team2 = winner THEN 1 ELSE 0 END AS win
    FROM matches
), team_wins AS (
    SELECT team, season, COUNT(*) AS matches, SUM(win) AS wins
    FROM team_matches
    GROUP BY team, season
), batting_rates AS (
    SELECT
        batting_team AS team,
        season,
        SUM(total_runs) AS runs_scored,
        SUM(CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END) AS legal_balls
    FROM deliveries
    GROUP BY batting_team, season
), bowling_wickets AS (
    SELECT
        bowling_team AS team,
        season,
        SUM(CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END) AS wickets
    FROM deliveries
    GROUP BY bowling_team, season
), combined AS (
    SELECT
        w.team,
        w.season,
        w.matches,
        w.wins,
        100.0 * w.wins / NULLIF(w.matches, 0) AS win_pct,
        6.0 * b.runs_scored / NULLIF(b.legal_balls, 0) AS runs_per_over,
        bw.wickets * 1.0 / NULLIF(w.matches, 0) AS wickets_per_match
    FROM team_wins w
    LEFT JOIN batting_rates b ON b.team = w.team AND b.season = w.season
    LEFT JOIN bowling_wickets bw ON bw.team = w.team AND bw.season = w.season
)
SELECT
    team,
    season,
    matches,
    wins,
    win_pct,
    runs_per_over,
    wickets_per_match,
    win_pct - LAG(win_pct) OVER (PARTITION BY team ORDER BY season) AS win_pct_yoy,
    runs_per_over - LAG(runs_per_over) OVER (PARTITION BY team ORDER BY season) AS rpo_yoy,
    wickets_per_match - LAG(wickets_per_match) OVER (PARTITION BY team ORDER BY season) AS wickets_yoy
FROM combined
ORDER BY team, season;
