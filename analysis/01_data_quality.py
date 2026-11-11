import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

print("1) WHO opened the PRs: author types")
print(con.sql("""
    SELECT COALESCE(author_type, '(deleted account)') AS author_type,
           COUNT(*) AS prs
    FROM pull_requests
    GROUP BY 1 ORDER BY prs DESC
"""))

print("2) Bots hiding as regular users: 'bot' in the name but type = User")
print(con.sql("""
    SELECT author_login, COUNT(*) AS prs
    FROM pull_requests
    WHERE author_type = 'User' AND LOWER(author_login) LIKE '%bot%'
    GROUP BY 1 ORDER BY prs DESC
    LIMIT 15
"""))

print("3) WHEN: PRs opened per year")
print(con.sql("""
    SELECT YEAR(created_at) AS year, COUNT(*) AS prs
    FROM pull_requests
    GROUP BY 1 ORDER BY 1
"""))

print("4) CONSISTENCY: does state agree with the dates?")
print(con.sql("""
    SELECT state,
           COUNT(*)                                        AS prs,
           SUM(CASE WHEN merged_at IS NOT NULL THEN 1 END) AS has_merged_date,
           SUM(CASE WHEN closed_at IS NOT NULL THEN 1 END) AS has_closed_date
    FROM pull_requests
    GROUP BY 1
"""))

print("5) Impossible timelines: closed or merged BEFORE being opened")
print(con.sql("""
    SELECT COUNT(*) AS bad_rows
    FROM pull_requests
    WHERE closed_at < created_at OR merged_at < created_at
"""))

con.close()