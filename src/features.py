import duckdb

MAIN_DB = "data/mergemate.duckdb"
PASS2_DB = "data/pass2.duckdb"


def build_basic(con):
    """Groups A (size), D (author), F (timing, language). One row per newcomer PR."""
    con.execute("""
        CREATE OR REPLACE TABLE feat_basic AS
        SELECT
            n.repo, n.number, n.created_at, n.author_login, n.merged_30d,
            r.language_group,
            LN(1 + n.additions)      AS log_additions,
            LN(1 + n.deletions)      AS log_deletions,
            LN(1 + n.changed_files)  AS log_changed_files,
            DAYOFWEEK(n.created_at)  AS weekday,
            HOUR(n.created_at)       AS hour_utc,
            DATE_DIFF('day', d.author_created_at, n.created_at) AS account_age_days,
            (SELECT COUNT(*)
               FROM prs_clean p
              WHERE p.author_login = n.author_login
                AND p.repo <> n.repo
                AND p.created_at < n.created_at)                AS prior_prs_elsewhere
        FROM newcomer_prs n
        JOIN repos r ON r.full_name = n.repo
        LEFT JOIN p2.pr_details d USING (repo, number)
    """)


def build_content(con):
    """Groups B (files) and C (description)."""
    con.execute(r"""
        CREATE OR REPLACE TABLE feat_content AS
        WITH files AS (
            SELECT repo, number,
                   BOOL_OR(regexp_matches(LOWER(path),
                       '(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.(py|go|rs)$|\.(test|spec)\.[jt]sx?$|tests?\.java$'))
                                                                    AS touches_tests,
                   BOOL_AND(regexp_matches(LOWER(path), '\.(md|rst|txt|adoc)$|(^|/)docs?/'))
                                                                    AS docs_only,
                   BOOL_OR(regexp_matches(LOWER(path), '^\.github/|\.gitlab-ci|jenkinsfile|\.circleci/'))
                                                                    AS touches_ci,
                   BOOL_OR(regexp_matches(LOWER(path),
                       '(^|/)(package(-lock)?\.json|yarn\.lock|pnpm-lock\.yaml|cargo\.(toml|lock)|go\.(mod|sum)|pom\.xml|build\.gradle(\.kts)?|requirements[^/]*\.txt|pyproject\.toml|setup\.py|poetry\.lock|uv\.lock)$'))
                                                                    AS touches_deps,
                   COUNT(DISTINCT split_part(path, '/', 1))         AS n_top_dirs
            FROM p2.pr_files
            GROUP BY 1, 2
        )
        SELECT n.repo, n.number,
               COALESCE(f.touches_tests, FALSE)  AS touches_tests,
               COALESCE(f.docs_only, FALSE)      AS docs_only,
               COALESCE(f.touches_ci, FALSE)     AS touches_ci,
               COALESCE(f.touches_deps, FALSE)   AS touches_deps,
               COALESCE(f.n_top_dirs, 0)         AS n_top_dirs,
               LENGTH(COALESCE(d.title, ''))     AS title_len,
               LENGTH(COALESCE(d.body, ''))      AS body_len,
               (d.body IS NULL OR TRIM(d.body) = '')                            AS body_empty,
               regexp_matches(LOWER(COALESCE(d.body, '')),
                              '(fix(es|ed)?|close[sd]?|resolve[sd]?)\s+#\d+')   AS mentions_issue,
               regexp_matches(COALESCE(d.body, ''), '\[[xX]\]')                 AS has_checklist,
               regexp_matches(LOWER(COALESCE(d.title, '')),
                              '^(feat|fix|docs|chore|refactor|test|perf|ci|build|style)(\(.+\))?!?:')
                                                                                AS conventional_title
        FROM newcomer_prs n
        LEFT JOIN files f          USING (repo, number)
        LEFT JOIN p2.pr_details d  USING (repo, number)
    """)


def build_repo(con):
    """Group E: the repo's state AT THE MOMENT each newcomer PR was opened (point-in-time)."""
    con.execute("""
        CREATE OR REPLACE TABLE feat_repo AS
        WITH nc_hist AS (
            SELECT repo, created_at,
                   CASE WHEN merged_at IS NOT NULL
                         AND merged_at <= created_at + INTERVAL 30 DAY THEN 1 ELSE 0 END AS m30
            FROM prs_flagged WHERE is_newcomer
        ),
        merge_hist AS (
            SELECT n.repo, n.number,
                   COUNT(h.created_at) AS repo_recent_n,
                   AVG(h.m30)          AS raw_recent_rate
            FROM newcomer_prs n
            LEFT JOIN nc_hist h
              ON  h.repo = n.repo
              AND h.created_at >= n.created_at - INTERVAL 210 DAY
              AND h.created_at <  n.created_at - INTERVAL 30 DAY
            GROUP BY 1, 2
        ),
        repo_load AS (
            SELECT n.repo, n.number,
                   COUNT(CASE WHEN p.closed_at IS NULL OR p.closed_at > n.created_at THEN 1 END)
                                                                       AS repo_open_backlog,
                   COUNT(CASE WHEN p.created_at >= n.created_at - INTERVAL 30 DAY THEN 1 END)
                                                                       AS repo_prs_last_30d
            FROM newcomer_prs n
            LEFT JOIN pull_requests p
              ON p.repo = n.repo AND p.created_at < n.created_at
            GROUP BY 1, 2
        )
        SELECT n.repo, n.number,
               DATE_DIFF('day', n.repo_first_pr_at, n.created_at)        AS repo_age_days,
               m.repo_recent_n,
               CASE WHEN m.repo_recent_n >= 5 THEN m.raw_recent_rate END AS repo_recent_merge_rate,
               l.repo_open_backlog,
               l.repo_prs_last_30d
        FROM newcomer_prs n
        JOIN merge_hist m USING (repo, number)
        JOIN repo_load  l USING (repo, number)
    """)

def build_final(con):
    """Join all feature groups, add transformed columns and the time-based split."""
    con.execute("""
        CREATE OR REPLACE TABLE features AS
        SELECT b.*,
               c.* EXCLUDE (repo, number),
               r.* EXCLUDE (repo, number),
               -- transforms for skewed counts (help the logistic baseline; trees don't need them)
               (b.prior_prs_elsewhere > 0)        AS has_prior_elsewhere,
               LN(1 + b.prior_prs_elsewhere)      AS log_prior_elsewhere,
               LN(1 + r.repo_open_backlog)        AS log_open_backlog,
               LN(1 + r.repo_prs_last_30d)        AS log_prs_last_30d,
               LN(1 + c.body_len)                 AS log_body_len,
               -- time-based split with a 30-day embargo before the test period
               CASE WHEN b.created_at >= '2026-01-01' THEN 'test'
                    WHEN b.created_at >= '2025-12-02' THEN 'gap'
                    ELSE 'train' END              AS split
        FROM feat_basic   b
        JOIN feat_content c USING (repo, number)
        JOIN feat_repo    r USING (repo, number)
    """)
    con.execute("COPY features TO 'data/features.parquet' (FORMAT PARQUET)")
if __name__ == "__main__":
    con = duckdb.connect(MAIN_DB)
    con.execute(f"ATTACH '{PASS2_DB}' AS p2 (READ_ONLY)")

    build_basic(con)
    build_content(con)
    build_repo(con)
    build_final(con)

    print("Final features table:")
    print(con.sql("""
        SELECT split, COUNT(*) AS rows, ROUND(100 * AVG(merged_30d), 1) AS pct_merged_30d,
               MIN(created_at)::DATE AS first_pr, MAX(created_at)::DATE AS last_pr
        FROM features GROUP BY 1 ORDER BY first_pr
    """).fetchdf().to_string())

    n_cols = len(con.execute("SELECT * FROM features LIMIT 0").description)
    print(f"\nColumns: {n_cols} | Saved to data/features.parquet")
    con.close()