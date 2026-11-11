import duckdb
from github_client import graphql
import time

MAIN_DB = "data/mergemate.duckdb"
PASS2_DB = "data/pass2.duckdb"
BATCH_SIZE = 25

FRAGMENT = """
fragment PRDetails on PullRequest {
  number
  title
  body
  author { ... on User { createdAt } }                         # account creation date
  closingIssuesReferences(first: 1) { totalCount }           # linked issues
  files(first: 100)   { nodes { path additions deletions } }
  comments(first: 10) { nodes { author { login } createdAt } }
  reviews(first: 10)  { nodes { author { login } createdAt state } }
}
"""

def build_query(numbers):
    """One query asking for several PRs of one repo, each under an alias like p17."""
    aliases = "\n".join(f"    p{n}: pullRequest(number: {n}) {{ ...PRDetails }}" for n in numbers)
    return f"""
query($owner: String!, $name: String!) {{
  repository(owner: $owner, name: $name) {{
{aliases}
  }}
  rateLimit {{ cost remaining }}
}}
{FRAGMENT}
"""

def load_targets():
    """The newcomer PRs we need details for, grouped by repo."""
    con = duckdb.connect(MAIN_DB, read_only=True)
    rows = con.execute("SELECT repo, number FROM newcomer_prs ORDER BY repo, number").fetchall()
    con.close()
    targets = {}
    for repo, number in rows:
        targets.setdefault(repo, []).append(number)
    return targets
def ts(value):
    return value.replace("T", " ").rstrip("Z") if value else None

def create_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS pr_details (
        repo VARCHAR, number INTEGER, title VARCHAR, body VARCHAR,
        author_created_at TIMESTAMP, linked_issues INTEGER,
        PRIMARY KEY (repo, number))""")
    con.execute("""CREATE TABLE IF NOT EXISTS pr_files (
        repo VARCHAR, number INTEGER, path VARCHAR, additions INTEGER, deletions INTEGER)""")
    con.execute("""CREATE TABLE IF NOT EXISTS pr_events (
        repo VARCHAR, number INTEGER, kind VARCHAR, actor_login VARCHAR,
        created_at TIMESTAMP, review_state VARCHAR)""")

def save_pr(con, repo, number, pr):
    """Save one PR's details, files and events. Safe to repeat (idempotent)."""
    con.execute("DELETE FROM pr_files  WHERE repo = ? AND number = ?", [repo, number])
    con.execute("DELETE FROM pr_events WHERE repo = ? AND number = ?", [repo, number])

    if pr is None:   # PR no longer reachable: record it so we don't retry forever
        con.execute("INSERT OR REPLACE INTO pr_details VALUES (?, ?, NULL, NULL, NULL, NULL)", [repo, number])
        return

    author = pr["author"] or {}
    con.execute("INSERT OR REPLACE INTO pr_details VALUES (?, ?, ?, ?, ?, ?)",
                [repo, number, pr["title"], pr["body"], ts(author.get("createdAt")),
                 pr["closingIssuesReferences"]["totalCount"]])

    files = [(repo, number, f["path"], f["additions"], f["deletions"]) for f in pr["files"]["nodes"]]
    if files:
        con.executemany("INSERT INTO pr_files VALUES (?, ?, ?, ?, ?)", files)

    events = []
    for c in pr["comments"]["nodes"]:
        events.append((repo, number, "comment", (c["author"] or {}).get("login"), ts(c["createdAt"]), None))
    for r in pr["reviews"]["nodes"]:
        events.append((repo, number, "review", (r["author"] or {}).get("login"), ts(r["createdAt"]), r["state"]))
    if events:
        con.executemany("INSERT INTO pr_events VALUES (?, ?, ?, ?, ?, ?)", events)
if __name__ == "__main__":
    targets = load_targets()
    con = duckdb.connect(PASS2_DB)
    create_tables(con)

    done = {(r, n) for r, n in con.execute("SELECT repo, number FROM pr_details").fetchall()}
    total = sum(len(v) for v in targets.values())
    print(f"Target PRs: {total:,} | already done: {len(done):,}")

    for i, (repo, numbers) in enumerate(targets.items(), start=1):
        todo = [n for n in numbers if (repo, n) not in done]
        if not todo:
            continue
        print(f"[{i}/{len(targets)}] {repo}: {len(todo)} PRs to fetch")
        owner, name = repo.split("/")

        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            data = graphql(build_query(batch), {"owner": owner, "name": name})
            if data is None or data["repository"] is None:
                print("  batch failed; will resume next run")
                continue
            for n in batch:
                save_pr(con, repo, n, data["repository"].get(f"p{n}"))
            time.sleep(1)

    saved = con.execute("SELECT COUNT(*) FROM pr_details").fetchall()[0][0]
    print(f"\nDone. PRs with details: {saved:,} of {total:,}")
    con.close()