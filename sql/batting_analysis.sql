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

-- name: Opening Partnership Success Rate
-- Business question: Which opening pairs produce 50+ stands most often (min 10 innings together)?
WITH base AS (
    SELECT
        match_id,
        inning,
        over,
        ball,
        batsman,
        non_striker,
        total_runs,
        is_wicket
    FROM deliveries
), ordered AS (
    SELECT
        *,
        (over - 1) * 6 + ball AS ball_index
    FROM base
), first_ball AS (
    SELECT
        match_id,
        inning,
        CASE WHEN batsman < non_striker THEN batsman ELSE non_striker END AS batter1,
        CASE WHEN batsman < non_striker THEN non_striker ELSE batsman END AS batter2
    FROM (
        SELECT
            match_id,
            inning,
            batsman,
            non_striker,
            ROW_NUMBER() OVER (PARTITION BY match_id, inning ORDER BY ball_index) AS rn
        FROM ordered
    )
    WHERE rn = 1
), first_wicket AS (
    SELECT match_id, inning, MIN(ball_index) AS wicket_ball
    FROM ordered
    WHERE is_wicket = 1
    GROUP BY match_id, inning
), partnership_runs AS (
    SELECT
        o.match_id,
        o.inning,
        SUM(o.total_runs) AS partnership_runs
    FROM ordered o
    LEFT JOIN first_wicket w ON w.match_id = o.match_id AND w.inning = o.inning
    WHERE w.wicket_ball IS NULL OR o.ball_index <= w.wicket_ball
    GROUP BY o.match_id, o.inning
), combined AS (
    SELECT f.batter1, f.batter2, p.partnership_runs
    FROM first_ball f
    JOIN partnership_runs p ON p.match_id = f.match_id AND p.inning = f.inning
)
SELECT
    batter1,
    batter2,
    COUNT(*) AS innings_together,
    AVG(partnership_runs) AS avg_partnership_runs,
    100.0 * SUM(CASE WHEN partnership_runs >= 50 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS fifty_plus_pct
FROM combined
GROUP BY batter1, batter2
HAVING COUNT(*) >= 10
ORDER BY fifty_plus_pct DESC, avg_partnership_runs DESC;

-- name: Early Wicket Impact
-- Business question: How does the first wicket timing affect final scores?
WITH innings_scores AS (
    SELECT match_id, inning, batting_team, SUM(total_runs) AS final_score
    FROM deliveries
    GROUP BY match_id, inning, batting_team
), first_wicket AS (
    SELECT match_id, inning, MIN(over) AS first_wicket_over
    FROM deliveries
    WHERE is_wicket = 1
    GROUP BY match_id, inning
)
SELECT
    CASE
        WHEN first_wicket_over BETWEEN 1 AND 3 THEN '1-3'
        WHEN first_wicket_over BETWEEN 4 AND 6 THEN '4-6'
        WHEN first_wicket_over >= 7 THEN '7+'
        ELSE 'No Wicket'
    END AS wicket_bucket,
    AVG(final_score) AS avg_final_score,
    COUNT(*) AS innings
FROM innings_scores s
LEFT JOIN first_wicket w ON w.match_id = s.match_id AND w.inning = s.inning
GROUP BY wicket_bucket
ORDER BY wicket_bucket;

-- name: Batsman Performance After Dot Ball Streaks
-- Business question: How many runs do batsmen score in the next 3 balls after 2+ dot balls?
WITH legal_balls AS (
    SELECT
        match_id,
        inning,
        batsman,
        over,
        ball,
        batsman_runs,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball
    FROM deliveries
), ordered AS (
    SELECT
        match_id,
        inning,
        batsman,
        over,
        ball,
        batsman_runs,
        ROW_NUMBER() OVER (PARTITION BY match_id, inning, batsman ORDER BY over, ball) AS rn
    FROM legal_balls
    WHERE legal_ball = 1
), flags AS (
    SELECT
        *,
        CASE WHEN batsman_runs = 0 THEN 1 ELSE 0 END AS dot_ball,
        LAG(CASE WHEN batsman_runs = 0 THEN 1 ELSE 0 END, 1) OVER (PARTITION BY match_id, inning, batsman ORDER BY rn) AS prev1,
        LAG(CASE WHEN batsman_runs = 0 THEN 1 ELSE 0 END, 2) OVER (PARTITION BY match_id, inning, batsman ORDER BY rn) AS prev2,
        LEAD(batsman_runs, 1) OVER (PARTITION BY match_id, inning, batsman ORDER BY rn) AS next1,
        LEAD(batsman_runs, 2) OVER (PARTITION BY match_id, inning, batsman ORDER BY rn) AS next2
    FROM ordered
), sequences AS (
    SELECT
        batsman,
        (batsman_runs + COALESCE(next1, 0) + COALESCE(next2, 0)) AS runs_next_3
    FROM flags
    WHERE prev1 = 1 AND prev2 = 1 AND batsman_runs > 0
)
SELECT
    batsman,
    COUNT(*) AS sequences,
    AVG(runs_next_3) AS avg_runs_next_3_balls
FROM sequences
GROUP BY batsman
HAVING COUNT(*) >= 10
ORDER BY avg_runs_next_3_balls DESC;

-- name: Score Bracket Frequency per Batsman
-- Business question: How often do batsmen score in each innings bracket (min 30 innings)?
WITH innings_runs AS (
    SELECT
        batsman,
        match_id,
        inning,
        SUM(batsman_runs) AS runs
    FROM deliveries
    GROUP BY batsman, match_id, inning
), eligible AS (
    SELECT batsman, COUNT(*) AS innings
    FROM innings_runs
    GROUP BY batsman
    HAVING COUNT(*) >= 30
), bucketed AS (
    SELECT
        i.batsman,
        e.innings,
        CASE
            WHEN i.runs BETWEEN 0 AND 10 THEN '0-10'
            WHEN i.runs BETWEEN 11 AND 30 THEN '11-30'
            WHEN i.runs BETWEEN 31 AND 50 THEN '31-50'
            WHEN i.runs BETWEEN 51 AND 99 THEN '50-99'
            ELSE '100+'
        END AS run_bucket
    FROM innings_runs i
    JOIN eligible e ON e.batsman = i.batsman
)
SELECT
    batsman,
    innings,
    100.0 * SUM(CASE WHEN run_bucket = '0-10' THEN 1 ELSE 0 END) / NULLIF(innings, 0) AS pct_0_10,
    100.0 * SUM(CASE WHEN run_bucket = '11-30' THEN 1 ELSE 0 END) / NULLIF(innings, 0) AS pct_11_30,
    100.0 * SUM(CASE WHEN run_bucket = '31-50' THEN 1 ELSE 0 END) / NULLIF(innings, 0) AS pct_31_50,
    100.0 * SUM(CASE WHEN run_bucket = '50-99' THEN 1 ELSE 0 END) / NULLIF(innings, 0) AS pct_50_99,
    100.0 * SUM(CASE WHEN run_bucket = '100+' THEN 1 ELSE 0 END) / NULLIF(innings, 0) AS pct_100_plus
FROM bucketed
GROUP BY batsman, innings
ORDER BY innings DESC;

-- name: Dismissal Type Breakdown per Batsman
-- Business question: What are the dismissal type shares for top 30 batsmen?
WITH top_batsmen AS (
    SELECT batsman, SUM(batsman_runs) AS total_runs
    FROM deliveries
    GROUP BY batsman
    ORDER BY total_runs DESC
    LIMIT 30
), dismissals AS (
    SELECT
        player_dismissed AS batsman,
        LOWER(COALESCE(dismissal_kind, '')) AS dismissal_kind
    FROM deliveries
    WHERE player_dismissed IS NOT NULL
), classified AS (
    SELECT
        d.batsman,
        CASE
            WHEN dismissal_kind LIKE '%bowled%' THEN 'bowled'
            WHEN dismissal_kind LIKE '%caught%' THEN 'caught'
            WHEN dismissal_kind LIKE '%lbw%' THEN 'lbw'
            WHEN dismissal_kind LIKE '%run out%' THEN 'run out'
            WHEN dismissal_kind LIKE '%stumped%' THEN 'stumped'
            ELSE 'other'
        END AS kind
    FROM dismissals d
    JOIN top_batsmen t ON t.batsman = d.batsman
)
SELECT
    batsman,
    100.0 * SUM(CASE WHEN kind = 'bowled' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS bowled_pct,
    100.0 * SUM(CASE WHEN kind = 'caught' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS caught_pct,
    100.0 * SUM(CASE WHEN kind = 'lbw' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS lbw_pct,
    100.0 * SUM(CASE WHEN kind = 'run out' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS run_out_pct,
    100.0 * SUM(CASE WHEN kind = 'stumped' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS stumped_pct,
    COUNT(*) AS dismissals
FROM classified
GROUP BY batsman
ORDER BY dismissals DESC;
