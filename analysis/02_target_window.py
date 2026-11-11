import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

print("Newcomer PRs from 2021 on: overall outcome")
print(con.sql("""
    SELECT state, COUNT(*) AS prs, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM prs_flagged
    WHERE is_newcomer AND created_at >= '2021-01-01'
    GROUP BY 1 ORDER BY prs DESC
"""))

print("Of the newcomer PRs that DID get merged: how long did it take?")
print(con.sql("""
    WITH merged AS (
        SELECT DATE_DIFF('day', created_at, merged_at) AS days_to_merge
        FROM prs_flagged
        WHERE is_newcomer AND created_at >= '2021-01-01' AND merged_at IS NOT NULL
    )
    SELECT
        COUNT(*)                                                      AS merged_prs,
        ROUND(100.0 * AVG(CASE WHEN days_to_merge <= 7  THEN 1 ELSE 0 END), 1) AS pct_within_7d,
        ROUND(100.0 * AVG(CASE WHEN days_to_merge <= 14 THEN 1 ELSE 0 END), 1) AS pct_within_14d,
        ROUND(100.0 * AVG(CASE WHEN days_to_merge <= 30 THEN 1 ELSE 0 END), 1) AS pct_within_30d,
        ROUND(100.0 * AVG(CASE WHEN days_to_merge <= 60 THEN 1 ELSE 0 END), 1) AS pct_within_60d,
        ROUND(100.0 * AVG(CASE WHEN days_to_merge <= 90 THEN 1 ELSE 0 END), 1) AS pct_within_90d,
        MEDIAN(days_to_merge)                                         AS median_days
    FROM merged
""").fetchdf().to_string())

con.close()