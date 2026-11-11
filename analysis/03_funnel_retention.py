import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

RETURN_DAYS = 90
RETURN_CUTOFF = "2026-08-12"     # decision must be at least 90 days before collection

print(con.sql(f"""
WITH first_pr AS (
    SELECT repo, author_login, created_at, closed_at, state
    FROM newcomer_prs
),
second_pr AS (
    SELECT repo, author_login, created_at AS second_at
    FROM prs_flagged
    WHERE author_pr_seq = 2
),
joined AS (
    SELECT f.*, s.second_at,
           CASE WHEN f.closed_at IS NULL     THEN 'no decision (still open)'
                WHEN f.state = 'MERGED'      THEN 'merged'
                ELSE                              'closed, not merged' END        AS outcome,
           -- opened a 2nd PR BEFORE the first was decided (so the outcome couldn't influence it)
           CASE WHEN s.second_at < COALESCE(f.closed_at, TIMESTAMP '9999-01-01')
                THEN 1 ELSE 0 END                                                 AS parallel_pr,
           -- came back AFTER learning the outcome, within 90 days of the decision
                      -- genuine return: a NEW contribution more than 1 day after the decision (excludes same-day retries)
           CASE WHEN s.second_at >  f.closed_at + INTERVAL 1 DAY
                 AND s.second_at <= f.closed_at + INTERVAL {RETURN_DAYS} DAY
                THEN 1 ELSE 0 END                                                 AS returned_after
    FROM first_pr f
    LEFT JOIN second_pr s USING (repo, author_login)
)
SELECT outcome,
       COUNT(*)                                                        AS newcomers,
       ROUND(100.0 * AVG(parallel_pr), 1)                              AS pct_parallel_pr,
       -- among those with NO parallel PR, and decided early enough to observe 90 days:
       ROUND(100.0 * AVG(CASE WHEN parallel_pr = 0 AND closed_at < '{RETURN_CUTOFF}'
                              THEN returned_after END), 1)             AS pct_returned_after_decision
FROM joined
GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\nHow soon after the decision did they come back? (returners only)")
print(con.sql(f"""
WITH first_pr AS (
    SELECT repo, author_login, closed_at, state
    FROM newcomer_prs
    WHERE closed_at IS NOT NULL AND closed_at < '{RETURN_CUTOFF}'
),
second_pr AS (
    SELECT repo, author_login, created_at AS second_at
    FROM prs_flagged WHERE author_pr_seq = 2
),
returners AS (
    SELECT CASE WHEN f.state = 'MERGED' THEN 'merged' ELSE 'closed, not merged' END AS outcome,
           DATE_DIFF('hour', f.closed_at, s.second_at) / 24.0 AS days_after
    FROM first_pr f JOIN second_pr s USING (repo, author_login)
    WHERE s.second_at > f.closed_at
      AND s.second_at <= f.closed_at + INTERVAL {RETURN_DAYS} DAY
)
SELECT outcome,
       COUNT(*)                                                         AS returners,
       ROUND(100.0 * AVG(CASE WHEN days_after <= 1 THEN 1 ELSE 0 END), 1) AS pct_within_1_day,
       ROUND(100.0 * AVG(CASE WHEN days_after <= 7 THEN 1 ELSE 0 END), 1) AS pct_within_7_days,
       ROUND(MEDIAN(days_after), 1)                                     AS median_days
FROM returners
GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())
con.close()