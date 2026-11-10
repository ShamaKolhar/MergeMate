import sys
import time
from datetime import datetime
from db import get_connection, create_pr_tables
from github_client import graphql

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        author { login __typename }
        authorAssociation
        createdAt closedAt mergedAt state
        additions deletions changedFiles
      }
    }
  }
  rateLimit { cost remaining }
}
"""

def ts(value):
    """GitHub sends '2019-01-23T10:15:00Z'; the database wants '2019-01-23 10:15:00'. None stays None."""
    return value.replace("T", " ").rstrip("Z") if value else None

def save_page(con, repo, nodes):
    """Insert (or overwrite) one page of PRs."""
    rows = []
    for pr in nodes:
        a = pr["author"]                                   # None if the account was deleted
        rows.append((
            repo, pr["number"],
            a["login"] if a else None, a["__typename"] if a else None,
            pr["authorAssociation"],
            ts(pr["createdAt"]), ts(pr["closedAt"]), ts(pr["mergedAt"]),
            pr["state"], pr["additions"], pr["deletions"], pr["changedFiles"],
        ))
    con.executemany("INSERT OR REPLACE INTO pull_requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

def collect_repo(con, repo):
    owner, name = repo.split("/")
    state = con.execute("SELECT last_cursor, done FROM collection_state WHERE repo = ?", [repo]).fetchone()
    if state and state[1]:
        print("  already done, skipping")
        return
    cursor = state[0] if state else None                  # resume from the bookmark, if there is one

    pages = 0
    while True:
        data = graphql(QUERY, {"owner": owner, "name": name, "cursor": cursor})
        if data is None or data["repository"] is None:
            print("  failed; will resume from the last bookmark next run")
            return

        prs = data["repository"]["pullRequests"]
        save_page(con, repo, prs["nodes"])
        cursor = prs["pageInfo"]["endCursor"]
        done = not prs["pageInfo"]["hasNextPage"]
        con.execute("INSERT OR REPLACE INTO collection_state VALUES (?, ?, ?, ?)",
                    [repo, cursor, done, datetime.now()])
        time.sleep(1)
        pages += 1
        if pages % 10 == 0:
            print(f"  {pages} pages | points left: {data['rateLimit']['remaining']}")
        if done:
            break

    count = con.execute("SELECT COUNT(*) FROM pull_requests WHERE repo = ?", [repo]).fetchone()[0]
    print(f"  done: {count} PRs stored")

if __name__ == "__main__":
    con = get_connection()
    create_pr_tables(con)
    repos = [r[0] for r in con.execute("SELECT full_name FROM repos ORDER BY pr_count").fetchall()]

    if len(sys.argv) > 1 and sys.argv[1] == "test":       # 'test' mode: only the smallest repo
        repos = repos[:1]

    for i, repo in enumerate(repos, start=1):
        print(f"[{i}/{len(repos)}] {repo}")
        collect_repo(con, repo)
    con.close()