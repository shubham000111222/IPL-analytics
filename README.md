# IPL Analytics (2008-2024)

A portfolio-ready analytics suite that transforms IPL match and ball-by-ball data into SQL insights, Python analysis, and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458)
![SQLite](https://img.shields.io/badge/SQLite-DB-003B57)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![CI-CD](https://github.com/shubham000111222/IPL-analytics/actions/workflows/ci-cd.yml/badge.svg)

## Problem Statement
Cricket boards and franchises need evidence-backed decisions on player selection, match strategy, and venue tactics. This project answers key questions on batting consistency, bowling pressure, and team strategy across IPL seasons. It combines match metadata with ball-by-ball events to reveal phase-wise performance, toss impact, and venue trends. The results support scouting, tactical planning, and opponent-specific preparation.

## Key Insights
- Batsmen with tenure > 5 seasons show 76.97% higher variance in scoring than shorter-tenure peers.
- Teams winning the toss chose to field 75.10% of the time post-2016.
- Death over economy below 9.29 correlates with a 27.11 percentage-point higher win rate.
- Powerplay scores for winning teams are 5.71 runs higher on average.
- Chasing success rate drops to 12.59% when required run rate exceeds 14.
- Arun Jaitley Stadium, Delhi has the highest first-innings average of 199.06 runs.

## Advanced Insights
- Anchors: KD Karthik, MS Dhoni, RV Uthappa, AT Rayudu, RA Jadeja, AM Rahane, MK Pandey, G Gambhir.
- Aggressors: RG Sharma, V Kohli, S Dhawan, SK Raina, DA Warner, KA Pollard, AB de Villiers, SV Samson.
- Win probability model accuracy: 0.647 (top features: pp_wickets_lost -0.56, powerplay_score +0.29, batting_first -0.22).
- Score predictor R2: 0.502.
- Most lopsided head-to-head record: Chennai Super Kings vs Sunrisers Hyderabad (71.4% over 21 matches).
- Biggest upset in IPL history: 2014 - Delhi Capitals over Kolkata Knight Riders at Dubai International Cricket Stadium (54.5 pp gap, margin 4).

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Data | Pandas, NumPy, scikit-learn |
| SQL | SQLite |
| Viz | Matplotlib, Seaborn, Plotly |
| App | Streamlit |

## Project Structure

```
ipl-analytics/
├── data/
│   └── raw/
│       ├── matches.csv
│       └── deliveries.csv
├── sql/
│   ├── schema.sql
│   ├── batting_analysis.sql
│   ├── bowling_analysis.sql
│   └── team_strategy.sql
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── analysis/
│   │   ├── batting.py
│   │   ├── bowling.py
│   │   └── team_strategy.py
│   └── viz/
│       └── plots.py
├── dashboard/
│   └── app.py
├── results/
│   └── (generated plots go here)
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## SQL Highlights

```sql
-- Orange Cap Race by Season
WITH runs_by_season AS (
    SELECT season, batsman, SUM(batsman_runs) AS total_runs
    FROM deliveries
    GROUP BY season, batsman
), ranked AS (
    SELECT season, batsman, total_runs,
           ROW_NUMBER() OVER (PARTITION BY season ORDER BY total_runs DESC) AS rn
    FROM runs_by_season
)
SELECT season, batsman, total_runs
FROM ranked
WHERE rn = 1
ORDER BY season;
```

```sql
-- Death Over Specialists
WITH base AS (
    SELECT
        bowler,
        total_runs - COALESCE(bye_runs, 0) - COALESCE(legbye_runs, 0) - COALESCE(penalty_runs, 0) AS runs_conceded,
        CASE WHEN COALESCE(wide_runs, 0) = 0 AND COALESCE(noball_runs, 0) = 0 THEN 1 ELSE 0 END AS legal_ball,
        CASE WHEN total_runs = 0 THEN 1 ELSE 0 END AS dot_ball,
        CASE WHEN is_wicket = 1 THEN 1 ELSE 0 END AS wicket
    FROM deliveries
    WHERE over BETWEEN 16 AND 20
)
SELECT bowler,
       6.0 * SUM(runs_conceded) / NULLIF(SUM(legal_ball), 0) AS death_economy,
       SUM(wicket) AS wickets,
       100.0 * SUM(dot_ball) / NULLIF(SUM(legal_ball), 0) AS dot_ball_pct
FROM base
GROUP BY bowler
HAVING SUM(legal_ball) / 6.0 >= 20
ORDER BY death_economy ASC;
```

```sql
-- Win Probability by Run Rate
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
)
SELECT *
FROM over_cum;
```

## Visualisations

![Top 15 Run Scorers](results/top_run_scorers.png)
![Top 15 Wicket Takers](results/top_wicket_takers.png)
![Strike Rate by Phase Heatmap](results/strike_rate_phase_heatmap.png)
![Economy Rate by Phase Heatmap](results/economy_phase_heatmap.png)
![Toss Win to Match Win Rate](results/toss_win_match_win.png)
![Venue Run Rates](results/venue_run_rates.png)
![Season Trends](results/season_trends.png)
![IPL Dynasty Chart](results/ipl_dynasty_chart.png)
![Batsman Consistency Scatter](results/batsman_consistency_scatter.png)
![Death Over Specialists](results/death_over_specialists.png)

## Visualisation Gallery

<table>
    <tr>
        <td><img src="results/top_run_scorers.png" width="400"/></td>
        <td><img src="results/top_wicket_takers.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/strike_rate_phase_heatmap.png" width="400"/></td>
        <td><img src="results/economy_phase_heatmap.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/toss_win_match_win.png" width="400"/></td>
        <td><img src="results/venue_run_rates.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/season_trends.png" width="400"/></td>
        <td><img src="results/ipl_dynasty_chart.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/batsman_consistency_scatter.png" width="400"/></td>
        <td><img src="results/death_over_specialists.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/batting_clusters.png" width="400"/></td>
        <td><img src="results/matchup_matrix.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/chasing_success.png" width="400"/></td>
        <td><img src="results/toss_trend.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/run_rate_pattern.png" width="400"/></td>
        <td><img src="results/win_factors.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/score_predictor.png" width="400"/></td>
        <td><img src="results/player_radar.png" width="400"/></td>
    </tr>
    <tr>
        <td><img src="results/wicket_types.png" width="400"/></td>
        <td><img src="results/h2h_matrix.png" width="400"/></td>
    </tr>
</table>

## Methodology
- Anchor vs Aggressor clusters use average runs, strike rate, dot-ball%, and boundary% with KMeans (2 clusters).
- Win probability model uses a logistic regression with powerplay score, wickets lost, toss, batting order, and venue averages.
- Limitations: data gaps, no ball-tracking data, and no player auction price context.

## How to Run

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Place the Kaggle CSVs in data/raw/
   - matches.csv
   - deliveries.csv
3. Load and clean data
   ```bash
   python src/data/load_data.py
   ```
4. Run analyses
   ```bash
   python src/analysis/batting.py
   python src/analysis/bowling.py
   python src/analysis/team_strategy.py
   ```
5. Generate plots
   ```bash
   python src/viz/plots.py
   ```
6. Launch dashboard
   ```bash
   streamlit run dashboard/app.py
   ```

## Deployment Guide

### Streamlit Community Cloud (recommended)
1. Push the repo to GitHub (already done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app" and set:
    - Repository: shubham000111222/IPL-analytics
    - Branch: main
    - Main file path: [dashboard/app.py](dashboard/app.py)
4. Click "Deploy".

### Common Deploy Checks
- Ensure dependencies are listed in [requirements.txt](requirements.txt).
- Keep the CSVs in [data/raw](data/raw) so the app can load them offline.
- If the app fails on first run, restart it after the initial build.

## CI/CD Pipeline
- Workflow file: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)
- Trigger: push/PR to main and manual dispatch.
- CI stage:
    - Installs dependencies from [requirements.txt](requirements.txt).
    - Performs syntax checks with compileall.
    - Rebuilds SQLite DB from [data/raw](data/raw).
    - Runs analysis scripts and regenerates plots.
    - Uploads [data/ipl.db](data/ipl.db) and [results](results) images as workflow artifacts.
- CD stage (main branch pushes only):
    - Downloads generated artifacts.
    - Publishes or updates a GitHub release tagged latest with the refreshed DB and plots.

## Dataset
Kaggle IPL dataset (2008-2024/2025): https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020/data

## Notes
- Update the Key Insights section after running the analysis scripts to insert the real numbers.
- Results plots are committed to the repo for quick viewing.


## Author

**Shubham Kumar** · NIT Delhi, CSE (3rd Year)  
[GitHub](https://github.com/shubham000111222) · [Portfolio](https://data-science-portfolio-three-olive.vercel.app)
[LinkedIn](https://linkedin.com/in/shubham-kumar-288b7437b)