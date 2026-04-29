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

-- name: Pressure Building Before Wickets
-- Business question: Which bowlers build the most dot-ball pressure before wickets? (min 30 wickets)
WITH legal_balls AS (
    SELECT
        match_id,
        inning,
        bowler,
        over,
        ball,
        total_runs,
        dismissal_kind,
        is_wicket,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball
    FROM deliveries
), ordered AS (
    SELECT
        match_id,
        inning,
        bowler,
        over,
        ball,
        total_runs,
        dismissal_kind,
        is_wicket,
        CASE WHEN total_runs = 0 THEN 1 ELSE 0 END AS dot_ball,
        SUM(CASE WHEN total_runs = 0 THEN 0 ELSE 1 END)
            OVER (PARTITION BY match_id, inning, bowler ORDER BY over, ball) AS dot_group
    FROM legal_balls
    WHERE legal_ball = 1
), dot_streaks AS (
    SELECT
        *,
        CASE WHEN dot_ball = 1 THEN COUNT(*) OVER (
            PARTITION BY match_id, inning, bowler, dot_group
        ) ELSE 0 END AS dot_streak_len
    FROM ordered
), wicket_context AS (
    SELECT
        *,
        LAG(dot_streak_len) OVER (PARTITION BY match_id, inning, bowler ORDER BY over, ball) AS dots_before_wicket
    FROM dot_streaks
)
SELECT
    bowler,
    AVG(COALESCE(dots_before_wicket, 0)) AS avg_dots_before_wicket,
    COUNT(*) AS wickets
FROM wicket_context
WHERE is_wicket = 1
  AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
GROUP BY bowler
HAVING COUNT(*) >= 30
ORDER BY avg_dots_before_wicket DESC;

-- name: Discipline Metric
-- Business question: Which bowlers concede the fewest wides/no-balls? (min 200 balls)
WITH base AS (
    SELECT
        bowler,
        CASE WHEN COALESCE(wide_runs, 0) > 0 THEN 1 ELSE 0 END AS wide_ball,
        CASE WHEN COALESCE(noball_runs, 0) > 0 THEN 1 ELSE 0 END AS noball_ball
    FROM deliveries
)
SELECT
    bowler,
    100.0 * SUM(wide_ball + noball_ball) / NULLIF(COUNT(*), 0) AS discipline_score,
    COUNT(*) AS total_balls
FROM base
GROUP BY bowler
HAVING COUNT(*) >= 200
ORDER BY discipline_score ASC;

-- name: Bowler Wicket Timing by Over
-- Business question: Which overs do bowlers take most wickets in?
SELECT
    bowler,
    over AS over_number,
    COUNT(*) AS wickets
FROM deliveries
WHERE is_wicket = 1
  AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
GROUP BY bowler, over
ORDER BY bowler, over_number;

-- name: Most Expensive Single Overs
-- Business question: Which overs have been the most expensive historically?
WITH over_batsman AS (
    SELECT
        match_id,
        inning,
        over,
        bowler,
        batsman,
        SUM(
            total_runs
            - COALESCE(bye_runs, 0)
            - COALESCE(legbye_runs, 0)
            - COALESCE(penalty_runs, 0)
        ) AS runs_conceded,
        SUM(batsman_runs) AS batsman_runs
    FROM deliveries
    GROUP BY match_id, inning, over, bowler, batsman
), over_totals AS (
    SELECT match_id, inning, over, bowler, SUM(runs_conceded) AS over_runs
    FROM over_batsman
    GROUP BY match_id, inning, over, bowler
), top_batsman AS (
    SELECT
        match_id,
        inning,
        over,
        bowler,
        batsman,
        batsman_runs,
        ROW_NUMBER() OVER (
            PARTITION BY match_id, inning, over, bowler
            ORDER BY batsman_runs DESC
        ) AS rn
    FROM over_batsman
)
SELECT
    o.match_id,
    o.inning,
    o.over,
    o.bowler,
    t.batsman,
    o.over_runs AS runs_conceded
FROM over_totals o
LEFT JOIN top_batsman t
    ON t.match_id = o.match_id
    AND t.inning = o.inning
    AND t.over = o.over
    AND t.bowler = o.bowler
    AND t.rn = 1
ORDER BY o.over_runs DESC
LIMIT 30;

-- name: Powerplay vs Death Specialist Split
-- Business question: Compare bowler economy and wickets in powerplay vs death overs.
WITH base AS (
    SELECT
        bowler,
        over,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball,
        CASE
            WHEN is_wicket = 1
             AND COALESCE(dismissal_kind, '') NOT IN ('run out', 'retired hurt', 'retired out', 'obstructing the field')
            THEN 1 ELSE 0 END AS wicket
    FROM deliveries
), phase AS (
    SELECT
        bowler,
        SUM(CASE WHEN over BETWEEN 1 AND 6 THEN runs_conceded ELSE 0 END) AS pp_runs,
        SUM(CASE WHEN over BETWEEN 1 AND 6 THEN legal_ball ELSE 0 END) AS pp_balls,
        SUM(CASE WHEN over BETWEEN 1 AND 6 THEN wicket ELSE 0 END) AS pp_wickets,
        SUM(CASE WHEN over BETWEEN 16 AND 20 THEN runs_conceded ELSE 0 END) AS death_runs,
        SUM(CASE WHEN over BETWEEN 16 AND 20 THEN legal_ball ELSE 0 END) AS death_balls,
        SUM(CASE WHEN over BETWEEN 16 AND 20 THEN wicket ELSE 0 END) AS death_wickets
    FROM base
    GROUP BY bowler
)
SELECT
    bowler,
    6.0 * pp_runs / NULLIF(pp_balls, 0) AS powerplay_economy,
    6.0 * death_runs / NULLIF(death_balls, 0) AS death_economy,
    pp_wickets AS powerplay_wickets,
    death_wickets,
    pp_balls / 6.0 AS powerplay_overs,
    death_balls / 6.0 AS death_overs
FROM phase
WHERE pp_balls / 6.0 >= 20
  AND death_balls / 6.0 >= 20
ORDER BY powerplay_economy ASC, death_economy ASC;
