import duckdb

DB_PATH = "data/mergemate.duckdb"

def get_connection():
    """Open (or create) the project database."""
    return duckdb.connect(DB_PATH)

def create_pr_tables(con):
    """Create the Pass 1 tables if they don't exist yet."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS pull_requests (
            repo                VARCHAR,
            number              INTEGER,
            author_login        VARCHAR,      -- NULL if the account was deleted
            author_type         VARCHAR,      -- 'User', 'Bot', ...
            author_association  VARCHAR,      -- stored, but NEVER used as a model feature (leaks)
            created_at          TIMESTAMP,
            closed_at           TIMESTAMP,
            merged_at           TIMESTAMP,    -- NULL means not merged
            state               VARCHAR,      -- OPEN / CLOSED / MERGED
            additions           INTEGER,
            deletions           INTEGER,
            changed_files       INTEGER,
            PRIMARY KEY (repo, number)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS collection_state (
            repo         VARCHAR PRIMARY KEY,
            last_cursor  VARCHAR,             -- bookmark to resume from
            done         BOOLEAN,
            updated_at   TIMESTAMP
        )
    """)

if __name__ == "__main__":
    con = get_connection()

    con.execute("""
        CREATE OR REPLACE TABLE repos AS
        SELECT
            full_name,
            "group"                       AS language_group,
            CAST(pr_count AS INTEGER)     AS pr_count,
            overall_merge_rate,
            outsider_merge_rate
        FROM read_csv_auto('repos.csv')
    """)

    
    print(con.sql("SELECT language_group, COUNT(*) AS repos, SUM(pr_count) AS total_prs "
                  "FROM repos GROUP BY language_group ORDER BY language_group"))
    print(con.sql("SELECT SUM(pr_count) AS all_prs FROM repos"))
    create_pr_tables(con)
    print(con.sql("SHOW TABLES"))
    con.close()