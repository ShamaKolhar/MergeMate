from db import get_connection

EXTRA_BOTS = ["tripleabuilderbot", "svcnemo-autobot", "fossabot",
              "OssSecurityBot", "RepoBirdBot", "WSandboxedOCCodeBot"]
TARGET_WINDOW_DAYS = 30
WINDOW_START = "2021-01-01"
WINDOW_END = "2026-10-11"
EARLY_REPO_DAYS = 90
def create_clean_view(con):
    bot_list = ", ".join(f"'{name}'" for name in EXTRA_BOTS)
    con.execute(f"""
        CREATE OR REPLACE VIEW prs_clean AS
        SELECT *
        FROM pull_requests
        WHERE author_type = 'User'                                    -- C1: no Bot accounts
          AND author_login IS NOT NULL                                -- C2: no deleted accounts
          AND NOT regexp_matches(LOWER(author_login), '-(ro)?bot$')   -- C3a: hyphen rule
          AND author_login NOT IN ({bot_list})                        -- C3b: extra bot list
          AND COALESCE(closed_at >= created_at, TRUE)                 -- C4: no impossible timelines
          AND COALESCE(merged_at >= created_at, TRUE)
    """)

def create_flagged_view(con):
    """Add each author's PR sequence number per repo; seq = 1 is their first PR (newcomer)."""
    con.execute("""
        CREATE OR REPLACE VIEW prs_flagged AS
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY repo, author_login
                   ORDER BY created_at, number          -- number breaks ties if two PRs share a timestamp
               ) AS author_pr_seq,
               (author_pr_seq = 1) AS is_newcomer
        FROM prs_clean
    """)
def create_newcomer_view(con):
    """One row per real newcomer PR in the modelling window, with the 30-day merge target."""
    con.execute(f"""
        CREATE OR REPLACE VIEW newcomer_prs AS
        WITH repo_start AS (
            SELECT repo, MIN(created_at) AS repo_first_pr_at
            FROM pull_requests
            GROUP BY repo
        )
        SELECT p.*,
               r.repo_first_pr_at,                                   -- kept for a 'repo age' feature later
               CASE WHEN p.merged_at IS NOT NULL
                         AND p.merged_at <= p.created_at + INTERVAL {TARGET_WINDOW_DAYS} DAY
                    THEN 1 ELSE 0 END AS merged_30d
        FROM prs_flagged p
        JOIN repo_start r USING (repo)
        WHERE p.is_newcomer
          AND p.created_at >= '{WINDOW_START}'
          AND p.created_at <  '{WINDOW_END}'
          AND p.created_at >= r.repo_first_pr_at + INTERVAL {EARLY_REPO_DAYS} DAY   -- decision A
    """)
if __name__ == "__main__":
    con = get_connection()
    create_clean_view(con)
    create_flagged_view(con)
    create_newcomer_view(con)

    raw = con.execute("SELECT COUNT(*) FROM pull_requests").fetchall()[0][0]
    clean = con.execute("SELECT COUNT(*) FROM prs_clean").fetchall()[0][0]
    print(f"Raw PRs:   {raw:,}")
    print(f"Clean PRs: {clean:,}")
    print(f"Removed:   {raw - clean:,} ({(raw - clean) / raw:.1%})")

    print("\nNewcomer PRs vs all PRs:")
    print(con.sql("""
        SELECT COUNT(*)                                        AS all_prs,
               SUM(CASE WHEN is_newcomer THEN 1 ELSE 0 END)    AS newcomer_prs,
               COUNT(DISTINCT repo || '/' || author_login)     AS distinct_repo_authors
        FROM prs_flagged
    """))

    print("Newcomer PRs per year:")
    print(con.sql("""
        SELECT YEAR(created_at) AS year, COUNT(*) AS newcomer_prs
        FROM prs_flagged
        WHERE is_newcomer
        GROUP BY 1 ORDER BY 1
    """))
    print("\nModelling base (newcomer PRs, 2021 to 10 Oct 2026):")
    print(con.sql("""
        SELECT COUNT(*)                       AS rows,
               SUM(merged_30d)                AS merged_30d,
               ROUND(100.0 * AVG(merged_30d), 1) AS pct_merged_30d
        FROM newcomer_prs
    """).fetchdf().to_string())

    print("\nTarget rate by year:")
    print(con.sql("""
        SELECT YEAR(created_at) AS year, COUNT(*) AS rows,
               ROUND(100.0 * AVG(merged_30d), 1) AS pct_merged_30d
        FROM newcomer_prs
        GROUP BY 1 ORDER BY 1
    """).fetchdf().to_string())
    con.close()