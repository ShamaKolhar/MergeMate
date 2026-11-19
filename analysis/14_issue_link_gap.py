"""How often does the mentions_issue regex miss issues that GitHub says are linked?"""
import duckdb

con = duckdb.connect()
con.execute("ATTACH 'data/pass2.duckdb' AS p2 (READ_ONLY)")

base = """
    FROM 'data/features.parquet' f
    JOIN p2.pr_details d USING (repo, number)
"""

print(con.execute(f"""
    SELECT COUNT(*) AS prs,
           ROUND(AVG(CAST(f.mentions_issue AS INT)), 3)                                  AS regex_says_linked,
           ROUND(AVG(CAST(COALESCE(d.linked_issues, 0) > 0 AS INT)), 3)                  AS github_says_linked,
           ROUND(AVG(CAST(COALESCE(d.linked_issues, 0) > 0 AND NOT f.mentions_issue AS INT)), 3) AS regex_missed
    {base}
""").df().to_string(index=False))

print("\nMerge rate by what each method sees:")
print(con.execute(f"""
    SELECT f.mentions_issue                      AS regex_linked,
           COALESCE(d.linked_issues, 0) > 0      AS github_linked,
           COUNT(*)                              AS prs,
           ROUND(AVG(CAST(f.merged_30d AS INT)), 3) AS merge_rate
    {base}
    GROUP BY 1, 2 ORDER BY 1, 2
""").df().to_string(index=False))