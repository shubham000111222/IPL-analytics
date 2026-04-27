-- name: Purple Cap Race by Season
-- Business question: Who led each season in total wickets and won the Purple Cap?
WITH wickets_by_season AS (
    SELECT
        season,
        bowler,
        SUM(CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END) AS wickets
    FROM deliveries
    GROUP BY season, bowler
), ranked AS (
    SELECT
        season,
        bowler,
        wickets,
        ROW_NUMBER() OVER (PARTITION BY season ORDER BY wickets DESC) AS rn
    FROM wickets_by_season
)
SELECT season, bowler, wickets
FROM ranked
WHERE rn = 1
ORDER BY season;

-- name: Bowler Economy by Phase
-- Business question: Which bowlers control scoring best across powerplay, middle, and death overs?
WITH base AS (
    SELECT
        bowler,
        over,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball
    FROM deliveries
)
SELECT
    bowler,
    6.0 * SUM(CASE WHEN over BETWEEN 1 AND 6 THEN runs_conceded ELSE 0 END)
        / NULLIF(SUM(CASE WHEN over BETWEEN 1 AND 6 THEN legal_ball ELSE 0 END), 0) AS powerplay_economy,
    6.0 * SUM(CASE WHEN over BETWEEN 7 AND 15 THEN runs_conceded ELSE 0 END)
        / NULLIF(SUM(CASE WHEN over BETWEEN 7 AND 15 THEN legal_ball ELSE 0 END), 0) AS middle_economy,
    6.0 * SUM(CASE WHEN over BETWEEN 16 AND 20 THEN runs_conceded ELSE 0 END)
        / NULLIF(SUM(CASE WHEN over BETWEEN 16 AND 20 THEN legal_ball ELSE 0 END), 0) AS death_economy,
    SUM(legal_ball) AS balls
FROM base
GROUP BY bowler
HAVING SUM(legal_ball) >= 200
ORDER BY death_economy ASC;

-- name: Dot Ball Percentage
-- Business question: Which bowlers generate the most pressure through dot balls?
WITH base AS (
    SELECT
        bowler,
        total_runs AS runs_off_bat,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball,
        CASE WHEN total_runs = 0 THEN 1 ELSE 0 END AS dot_ball,
        CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END AS wicket
    FROM deliveries
)
SELECT
    bowler,
    100.0 * SUM(dot_ball) / NULLIF(SUM(legal_ball), 0) AS dot_ball_pct,
    6.0 * SUM(runs_conceded) / NULLIF(SUM(legal_ball), 0) AS economy,
    SUM(wicket) AS wickets,
    SUM(legal_ball) AS balls
FROM base
GROUP BY bowler
HAVING SUM(legal_ball) >= 200
ORDER BY dot_ball_pct DESC, economy ASC;

-- name: Death Over Specialists
-- Business question: Which bowlers excel in overs 17-20 under high-pressure death phases?
WITH base AS (
    SELECT
        bowler,
        over,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball,
        CASE WHEN total_runs = 0 THEN 1 ELSE 0 END AS dot_ball,
        CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END AS wicket
    FROM deliveries
    WHERE over BETWEEN 16 AND 20
)
SELECT
    bowler,
    6.0 * SUM(runs_conceded) / NULLIF(SUM(legal_ball), 0) AS death_economy,
    SUM(wicket) AS wickets,
    100.0 * SUM(dot_ball) / NULLIF(SUM(legal_ball), 0) AS dot_ball_pct,
    SUM(runs_conceded) * 1.0 / NULLIF(SUM(legal_ball) / 6.0, 0) AS avg_runs_per_over,
    SUM(legal_ball) / 6.0 AS overs
FROM base
GROUP BY bowler
HAVING SUM(legal_ball) / 6.0 >= 20
ORDER BY death_economy ASC, wickets DESC;

-- name: Bowler vs Left/Right Handed Batsmen
-- Business question: Do bowlers perform differently against left- and right-handed batters?
-- Note: Requires a column like batsman_hand or batting_hand in the deliveries table.
WITH base AS (
    SELECT
        bowler,
        batsman_hand AS hand,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball,
        CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END AS wicket
    FROM deliveries
    WHERE batsman_hand IS NOT NULL
)
SELECT
    bowler,
    hand,
    6.0 * SUM(runs_conceded) / NULLIF(SUM(legal_ball), 0) AS economy,
    SUM(wicket) AS wickets,
    SUM(legal_ball) AS balls
FROM base
GROUP BY bowler, hand
HAVING SUM(legal_ball) >= 100
ORDER BY economy ASC;
