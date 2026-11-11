import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

df = con.sql("""
    SELECT author_login, COUNT(*) AS prs
    FROM pull_requests
    WHERE author_type = 'User'
      AND regexp_matches(LOWER(author_login), '(ro)?bot$')
    GROUP BY 1
    ORDER BY prs DESC
""").fetchdf()

print(df.to_string())
print("\nTotal logins:", len(df), "| Total PRs:", df["prs"].sum())
con.close()