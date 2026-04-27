-- name: Orange Cap Race by Season
-- Business question: Who led each season in total runs and won the Orange Cap?
WITH runs_by_season AS (
    SELECT
        season,
        batsman,
        SUM(batsman_runs) AS total_runs
    FROM deliveries
    GROUP BY season, batsman
), ranked AS (
    SELECT
        season,
        batsman,
        total_runs,
        ROW_NUMBER() OVER (PARTITION BY season ORDER BY total_runs DESC) AS rn
    FROM runs_by_season
)
SELECT season, batsman, total_runs
FROM ranked
WHERE rn = 1
ORDER BY season;

-- name: Batsman Consistency Index
-- Business question: Which batsmen combine high scoring with low volatility over long careers?
WITH innings_runs AS (
    SELECT
        batsman,
        match_id,
        inning,
        SUM(batsman_runs) AS runs,
        SUM(CASE WHEN COALESCE(wide_runs, 0) = 0 THEN 1 ELSE 0 END) AS balls
    FROM deliveries
    GROUP BY batsman, match_id, inning
), batsman_agg AS (
    SELECT
        batsman,
        COUNT(*) AS innings,
        AVG(runs) AS avg_runs,
        (AVG(runs * runs) - AVG(runs) * AVG(runs)) AS var_runs,
        SUM(runs) AS total_runs,
        SUM(balls) AS total_balls
    FROM innings_runs
    GROUP BY batsman
    HAVING COUNT(*) >= 50
)
SELECT
    batsman,
    avg_runs,
    sqrt(CASE WHEN var_runs < 0 THEN 0 ELSE var_runs END) AS stddev_runs,
    avg_runs / NULLIF(sqrt(CASE WHEN var_runs < 0 THEN 0 ELSE var_runs END), 0) AS consistency_index,
    100.0 * total_runs / NULLIF(total_balls, 0) AS strike_rate,
    innings
FROM batsman_agg
ORDER BY consistency_index DESC, avg_runs DESC;

-- name: Strike Rate by Phase
-- Business question: How do top batsmen adjust scoring speed across innings phases?
WITH balls_faced AS (
    SELECT
        batsman,
        over,
        batsman_runs,
        CASE WHEN COALESCE(wide_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball
    FROM deliveries
), phase_agg AS (
    SELECT
        batsman,
        SUM(CASE WHEN over BETWEEN 1 AND 6 THEN batsman_runs ELSE 0 END) AS pp_runs,
        SUM(CASE WHEN over BETWEEN 7 AND 15 THEN batsman_runs ELSE 0 END) AS mid_runs,
        SUM(CASE WHEN over BETWEEN 16 AND 20 THEN batsman_runs ELSE 0 END) AS death_runs,
        SUM(CASE WHEN over BETWEEN 1 AND 6 THEN legal_ball ELSE 0 END) AS pp_balls,
        SUM(CASE WHEN over BETWEEN 7 AND 15 THEN legal_ball ELSE 0 END) AS mid_balls,
        SUM(CASE WHEN over BETWEEN 16 AND 20 THEN legal_ball ELSE 0 END) AS death_balls,
        SUM(legal_ball) AS total_balls
    FROM balls_faced
    GROUP BY batsman
    HAVING SUM(legal_ball) >= 200
)
SELECT
    batsman,
    100.0 * pp_runs / NULLIF(pp_balls, 0) AS powerplay_sr,
    100.0 * mid_runs / NULLIF(mid_balls, 0) AS middle_sr,
    100.0 * death_runs / NULLIF(death_balls, 0) AS death_sr,
    total_balls
FROM phase_agg
ORDER BY death_sr DESC;

-- name: Boundary Analysis
-- Business question: Which batsmen derive the largest share of runs from boundaries?
WITH innings_runs AS (
    SELECT
        batsman,
        match_id,
        inning,
        SUM(batsman_runs) AS runs
    FROM deliveries
    GROUP BY batsman, match_id, inning
), boundary_agg AS (
    SELECT
        d.batsman,
        SUM(CASE WHEN batsman_runs = 4 THEN 1 ELSE 0 END) AS fours,
        SUM(CASE WHEN batsman_runs = 6 THEN 1 ELSE 0 END) AS sixes,
        SUM(batsman_runs) AS total_runs
    FROM deliveries d
    GROUP BY d.batsman
), innings_count AS (
    SELECT batsman, COUNT(*) AS innings
    FROM innings_runs
    GROUP BY batsman
)
SELECT
    b.batsman,
    b.fours,
    b.sixes,
    100.0 * (4 * b.fours + 6 * b.sixes) / NULLIF(b.total_runs, 0) AS boundary_percentage,
    (b.fours + b.sixes) * 1.0 / NULLIF(i.innings, 0) AS avg_boundaries_per_innings,
    b.total_runs
FROM boundary_agg b
JOIN innings_count i ON i.batsman = b.batsman
ORDER BY boundary_percentage DESC, b.total_runs DESC;

-- name: Performance vs Specific Teams
-- Business question: Which batsmen consistently dominate specific opposition teams?
WITH innings_runs AS (
    SELECT
        batsman,
        bowling_team AS opposition,
        match_id,
        inning,
        SUM(batsman_runs) AS runs,
        SUM(CASE WHEN COALESCE(wide_runs, 0) = 0 THEN 1 ELSE 0 END) AS balls
    FROM deliveries
    GROUP BY batsman, opposition, match_id, inning
), agg AS (
    SELECT
        batsman,
        opposition,
        COUNT(*) AS innings,
        AVG(runs) AS avg_runs,
        SUM(runs) AS total_runs,
        SUM(balls) AS total_balls
    FROM innings_runs
    GROUP BY batsman, opposition
    HAVING COUNT(*) >= 3
)
SELECT
    batsman,
    opposition,
    avg_runs,
    100.0 * total_runs / NULLIF(total_balls, 0) AS strike_rate,
    innings
FROM agg
ORDER BY avg_runs DESC, strike_rate DESC;
