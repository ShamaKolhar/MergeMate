import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

print("1) By year: outcome mix and volume per repo")
print(con.sql("""
    SELECT YEAR(created_at)                                               AS year,
           COUNT(*)                                                       AS newcomer_prs,
           COUNT(DISTINCT repo)                                           AS active_repos,
           ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT repo), 1)                AS prs_per_repo,
           ROUND(100.0 * AVG(merged_30d), 1)                              AS pct_merged_30d,
           ROUND(100.0 * AVG(CASE WHEN state = 'CLOSED' THEN 1 ELSE 0 END), 1) AS pct_rejected,
           ROUND(100.0 * AVG(CASE WHEN state = 'OPEN'   THEN 1 ELSE 0 END), 1) AS pct_still_open
    FROM newcomer_prs
    GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\n2) Fixed panel: SAME repos, early (2021-24) vs late (2025-26)")
print(con.sql("""
    WITH per_repo AS (
        SELECT repo,
               CASE WHEN YEAR(created_at) <= 2024 THEN 'early' ELSE 'late' END AS period,
               COUNT(*)          AS prs,
               AVG(merged_30d)   AS merge_rate
        FROM newcomer_prs
        GROUP BY 1, 2
    ),
    panel AS (                                   -- repos with at least 20 newcomer PRs in BOTH periods
        SELECT repo FROM per_repo WHERE prs >= 20
        GROUP BY repo HAVING COUNT(*) = 2
    )
    SELECT period,
           COUNT(*)                               AS repos,
           SUM(prs)                               AS newcomer_prs,
           ROUND(100.0 * SUM(merge_rate * prs) / SUM(prs), 1) AS pct_merged_30d
    FROM per_repo
    WHERE repo IN (SELECT repo FROM panel)
    GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\n3) Fair comparison: status at DAY 30 for every PR (same window for all years)")
print(con.sql("""
    SELECT YEAR(created_at) AS year,
           COUNT(*)         AS newcomer_prs,
           ROUND(100.0 * AVG(merged_30d), 1)                                          AS pct_merged_by_30d,
           ROUND(100.0 * AVG(CASE WHEN state = 'CLOSED'
                                   AND closed_at <= created_at + INTERVAL 30 DAY
                                  THEN 1 ELSE 0 END), 1)                              AS pct_rejected_by_30d,
           ROUND(100.0 * AVG(CASE WHEN closed_at IS NULL
                                    OR closed_at > created_at + INTERVAL 30 DAY
                                  THEN 1 ELSE 0 END), 1)                              AS pct_undecided_at_30d
    FROM newcomer_prs
    GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())
con.close()