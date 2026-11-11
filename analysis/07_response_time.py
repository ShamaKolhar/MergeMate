import duckdb

BOT_LIST = ["coderabbitai", "github-actions", "sonarqubecloud", "welcome", "codecov",
            "codecov-commenter", "CLAassistant", "qodo-free-for-open-source-projects",
            "copy-pr-bot", "jetstack-bot", "netlify", "cert-manager-prow", "paul-the-alien",
            "trag-bot", "jabref-machine", "copilot-pull-request-reviewer", "dynamo-review-agent"]
bot_sql = ", ".join(f"'{b}'" for b in BOT_LIST)

con = duckdb.connect("data/mergemate.duckdb", read_only=True)
con.execute("ATTACH 'data/pass2.duckdb' AS p2 (READ_ONLY)")

BASE = f"""
WITH ev AS (                      -- every comment/review by someone other than the PR author
    SELECT n.repo, n.number, n.created_at AS pr_created_at, e.actor_login, e.kind, e.created_at AS ev_at
    FROM newcomer_prs n
    JOIN p2.pr_events e USING (repo, number)
    WHERE e.actor_login IS NOT NULL AND e.actor_login <> n.author_login
),
actor_stats AS (                  -- for rule 3: fast, high-volume accounts
    SELECT actor_login, COUNT(*) AS events,
           MEDIAN(DATE_DIFF('second', pr_created_at, ev_at)) / 60.0 AS median_min
    FROM ev GROUP BY 1
),
bots AS (
    SELECT actor_login FROM actor_stats
    WHERE actor_login IN ({bot_sql})                                        -- rule 1: explicit list
       OR regexp_matches(LOWER(actor_login), '(-(ro)?bot$|\\[bot\\]|-ci$|prow)')  -- rule 2: patterns
       OR (median_min < 5 AND events >= 50)                                -- rule 3: fast + high volume
),
first_human AS (
    SELECT repo, number, MIN(ev_at) AS first_human_at
    FROM ev WHERE actor_login NOT IN (SELECT actor_login FROM bots)
    GROUP BY 1, 2
),
pr AS (
    SELECT n.repo, n.number, n.created_at, n.merged_30d, f.first_human_at,
           DATE_DIFF('minute', n.created_at, f.first_human_at) / 60.0 AS hours_to_human
    FROM newcomer_prs n
    LEFT JOIN first_human f USING (repo, number)
)
"""

print("1) Human response: coverage and speed, by year")
print(con.sql(BASE + """
    SELECT YEAR(created_at) AS year, COUNT(*) AS prs,
           ROUND(100.0 * AVG(CASE WHEN hours_to_human <= 24*30 THEN 1 ELSE 0 END), 1) AS pct_human_reply_30d,
           ROUND(MEDIAN(hours_to_human), 1)                                            AS median_hours_to_human
    FROM pr GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\n2) Merge rate by time to first human response")
print(con.sql(BASE + """
    SELECT CASE WHEN hours_to_human IS NULL OR hours_to_human > 24*30 THEN '4) none within 30 days'
                WHEN hours_to_human <= 24     THEN '1) within 1 day'
                WHEN hours_to_human <= 24*7   THEN '2) 1 to 7 days'
                ELSE                               '3) 7 to 30 days' END AS response_bucket,
           COUNT(*)                            AS prs,
           ROUND(100.0 * AVG(merged_30d), 1)   AS pct_merged_30d
    FROM pr GROUP BY 1 ORDER BY 1
""").fetchdf().to_string())

print("\n3) Data-limit check: PRs that hit the 10-event cap with NO human found")
print(con.sql(BASE + """
    , caps AS (
        SELECT repo, number,
               SUM(CASE WHEN kind = 'comment' THEN 1 ELSE 0 END) AS n_comments,
               SUM(CASE WHEN kind = 'review'  THEN 1 ELSE 0 END) AS n_reviews
        FROM p2.pr_events GROUP BY 1, 2
    )
    SELECT COUNT(*) AS capped_prs_without_human,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM newcomer_prs), 2) AS pct_of_all
    FROM pr JOIN caps USING (repo, number)
    WHERE pr.first_human_at IS NULL AND (n_comments >= 10 OR n_reviews >= 10)
""").fetchdf().to_string())

con.close()