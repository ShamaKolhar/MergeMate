import duckdb

con = duckdb.connect("data/mergemate.duckdb", read_only=True)
print(con.sql("""
    SELECT author_login,
           COUNT(*)                    AS newcomer_rows,          -- repos where they were a "newcomer"
           MAX(prior_prs_elsewhere)    AS max_prior_elsewhere,
           ROUND(AVG(merged_30d), 2)   AS merge_rate
    FROM feat_basic
    GROUP BY 1
    ORDER BY max_prior_elsewhere DESC
    LIMIT 15
""").fetchdf().to_string())
con.close()