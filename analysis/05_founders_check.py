import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

print("Newcomer PRs opened in a repo's EARLY days (likely founders / core team)")
print(con.sql("""
    WITH repo_start AS (
        SELECT repo, MIN(created_at) AS first_pr_at
        FROM pull_requests
        GROUP BY repo
    )
    SELECT CASE WHEN n.created_at < r.first_pr_at + INTERVAL 90 DAY
                THEN 'within 90 days of repo start' ELSE 'later' END   AS timing,
           COUNT(*)                                                     AS newcomer_prs,
           ROUND(100.0 * AVG(n.merged_30d), 1)                          AS pct_merged_30d
    FROM newcomer_prs n
    JOIN repo_start r USING (repo)
    GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\nWhich repos started inside our modelling window (2021+)?")
print(con.sql("""
    SELECT repo, MIN(created_at)::DATE AS first_pr_date
    FROM pull_requests
    GROUP BY repo
    HAVING MIN(created_at) >= '2021-01-01'
    ORDER BY first_pr_date
""").fetchdf().to_string())

con.close()