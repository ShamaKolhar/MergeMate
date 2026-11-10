from github_client import graphql

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 5, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
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

data = graphql(QUERY, {"owner": "fastapi", "name": "fastapi", "cursor": None})

prs = data["repository"]["pullRequests"]
for pr in prs["nodes"]:
    author = pr["author"]["login"] if pr["author"] else "(deleted)"
    print(pr["number"], author, pr["author"]["__typename"] if pr["author"] else "-",
          pr["state"], pr["createdAt"][:10], f"+{pr['additions']}/-{pr['deletions']}")

print("\nHas next page:", prs["pageInfo"]["hasNextPage"])
print("Bookmark (endCursor):", prs["pageInfo"]["endCursor"])
print("This query cost:", data["rateLimit"]["cost"], "| points left:", data["rateLimit"]["remaining"])