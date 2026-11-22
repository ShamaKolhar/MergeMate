"""Fetch a live pull request from one of the 50 repos and build the model's features for it."""
import re
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from features import build_basic, build_content, build_repo, MAIN_DB
from github_client import graphql, get_with_retry
from model_config import FEATURES, BOOLEAN

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number title body createdAt additions deletions changedFiles state
      author { login ... on User { createdAt } }
      files(first: 100) { nodes { path additions deletions } }
    }
  }
}
"""


def parse_pr_url(url):
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
    if not m:
        raise ValueError("Expected a link like https://github.com/owner/repo/pull/123")
    return m.group(1), int(m.group(2))


def search_count(query):
    """How many issues/PRs match a GitHub search (only the count, not the results)."""
    r = get_with_retry("https://api.github.com/search/issues", params={"q": query, "per_page": 1})
    if r is None or r.status_code != 200:
        return None
    return r.json()["total_count"]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_pr(url):
    repo, number = parse_pr_url(url)
    owner, name = repo.split("/")
    data = graphql(QUERY, {"owner": owner, "name": name, "number": number})
    pr = (data or {}).get("repository", {}) and data["repository"]["pullRequest"]
    if not pr:
        raise ValueError(f"Could not fetch {repo}#{number}")

    created = datetime.fromisoformat(pr["createdAt"].replace("Z", "+00:00")).replace(tzinfo=None)
    author = pr["author"] or {}
    login = author.get("login")
    acct = author.get("createdAt")

    return {
        "repo": repo, "number": number, "title": pr["title"], "body": pr["body"],
        "state": pr["state"], "created_at": created, "author_login": login,
        "author_created_at": (datetime.fromisoformat(acct.replace("Z", "+00:00")).replace(tzinfo=None)
                              if acct else None),
        "additions": pr["additions"], "deletions": pr["deletions"],
        "changed_files": pr["changedFiles"], "files": pr["files"]["nodes"],
        # live counts: our stored data stops on 10 Nov
        "open_backlog_now": search_count(f"repo:{repo} is:pr is:open"),
        "prs_last_30d": search_count(
            f"repo:{repo} is:pr created:{iso(created - timedelta(days=30))}..{iso(created)}"),
        "author_prior_prs_here": search_count(
            f"repo:{repo} is:pr author:{login} created:<{iso(created)}") if login else None,
    }


def live_features(pr):
    """The 24 model features for a live PR, built with the same SQL as training."""
    con = duckdb.connect()                                   # in-memory scratch database
    con.execute(f"ATTACH '{MAIN_DB}' AS m (READ_ONLY)")
    for name in ["repos", "pull_requests", "prs_clean", "prs_flagged"]:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM m.{name}")

    groups = [g for (g,) in con.execute(
        "SELECT DISTINCT language_group FROM repos ORDER BY 1").fetchall()]
    lang = con.execute("SELECT language_group FROM repos WHERE full_name = ?",
                       [pr["repo"]]).fetchall()
    if not lang:
        raise ValueError(f"{pr['repo']} is not one of the 50 repositories this model covers")

    # The live PR, shaped like the training tables
    con.execute("CREATE SCHEMA p2")
    con.execute("""CREATE TABLE p2.pr_details AS SELECT ?::VARCHAR AS repo, ?::INTEGER AS number,
                   ?::VARCHAR AS title, ?::VARCHAR AS body, ?::TIMESTAMP AS author_created_at,
                   NULL::INTEGER AS linked_issues""",
                [pr["repo"], pr["number"], pr["title"], pr["body"], pr["author_created_at"]])
    con.execute("CREATE TABLE p2.pr_files (repo VARCHAR, number INTEGER, path VARCHAR, "
                "additions INTEGER, deletions INTEGER)")
    if pr["files"]:
        con.executemany("INSERT INTO p2.pr_files VALUES (?, ?, ?, ?, ?)",
                        [(pr["repo"], pr["number"], f["path"], f["additions"], f["deletions"])
                         for f in pr["files"]])
    con.execute("""CREATE TABLE newcomer_prs AS
                   SELECT ?::VARCHAR AS repo, ?::INTEGER AS number, ?::TIMESTAMP AS created_at,
                          ?::VARCHAR AS author_login, NULL::INTEGER AS merged_30d,
                          ?::INTEGER AS additions, ?::INTEGER AS deletions, ?::INTEGER AS changed_files,
                          (SELECT MIN(created_at) FROM pull_requests WHERE repo = ?) AS repo_first_pr_at""",
                [pr["repo"], pr["number"], pr["created_at"], pr["author_login"],
                 pr["additions"], pr["deletions"], pr["changed_files"], pr["repo"]])

    build_basic(con)
    build_content(con)
    build_repo(con)          # NOT build_final: it would overwrite data/features.parquet
    row = con.execute("""SELECT * FROM feat_basic b
                         JOIN feat_content c USING (repo, number)
                         JOIN feat_repo r USING (repo, number)""").df().iloc[0]
    con.close()

    # Same transforms as build_final + load_data, with the live counts for repo load
    f = {k: row[k] for k in ["log_additions", "log_deletions", "log_changed_files", "n_top_dirs",
                             "title_len", "repo_age_days", "repo_recent_merge_rate",
                             "hour_utc", "weekday", "touches_tests", "docs_only", "touches_ci",
                             "touches_deps", "body_empty", "mentions_issue", "has_checklist",
                             "conventional_title"]}
    f["log_account_age"] = np.log1p(row["account_age_days"])
    f["log_prior_elsewhere"] = np.log1p(row["prior_prs_elsewhere"])
    f["has_prior_elsewhere"] = row["prior_prs_elsewhere"] > 0
    f["log_body_len"] = np.log1p(row["body_len"])
    f["log_open_backlog"] = np.log1p(pr["open_backlog_now"])
    f["log_prs_last_30d"] = np.log1p(pr["prs_last_30d"])
    f["language_group"] = groups.index(lang[0][0])

    X = pd.DataFrame([f])[FEATURES]
    X[BOOLEAN] = X[BOOLEAN].astype(int)
    return X


if __name__ == "__main__":
    # Skew test: live features vs the stored training row for the same PR
    from model_config import load_data
    pr = fetch_pr("https://github.com/JabRef/jabref/pull/15777")
    live = live_features(pr).iloc[0]
    df = load_data()
    stored = df[(df["repo"] == "JabRef/jabref") & (df["number"] == 15777)][FEATURES].iloc[0]

    print(f"{'feature':24} {'stored':>10} {'live':>10}")
    for col in FEATURES:
        s, l = float(stored[col]), float(live[col])
        flag = "" if np.isclose(s, l, equal_nan=True) else "  <-- differs"
        print(f"{col:24} {s:10.3f} {l:10.3f}{flag}")