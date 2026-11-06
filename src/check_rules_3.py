import os
import pandas as pd
from github_client import get_with_retry

INSIDERS = {"OWNER", "MEMBER", "COLLABORATOR"}
OUT = "data/candidates_rules_5_6.csv"

def merge_rates(full_name):
    """From the 100 most recent closed PRs: overall merge rate, outsider merge rate, outsider count."""
    params = {"state": "closed", "sort": "created", "direction": "desc", "per_page": 100}
    r = get_with_retry(f"https://api.github.com/repos/{full_name}/pulls", params)
    if r is None or r.status_code != 200:
        return None, None, None

    # Drop bots and deleted accounts (user can be None)
    prs = [pr for pr in r.json()
           if pr["user"] and pr["user"]["type"] != "Bot"]
    if not prs:
        return None, None, None

    overall = sum(pr["merged_at"] is not None for pr in prs) / len(prs)

    outsiders = [pr for pr in prs if pr["author_association"] not in INSIDERS]
    if outsiders:
        outsider_rate = sum(pr["merged_at"] is not None for pr in outsiders) / len(outsiders)
    else:
        outsider_rate = 0.0

    return overall, outsider_rate, len(outsiders)

# Start from repos passing Rules 1-4
prev = pd.read_csv("data/candidates_rules_2_4.csv").drop_duplicates("full_name", keep="last")
pool = prev[(prev["contributing"] == True) & (prev["active"] == True)]

# Resume
if os.path.exists(OUT):
    done_names = set(pd.read_csv(OUT).dropna(subset=["overall_merge_rate"])["full_name"])
else:
    done_names = set()
    pd.DataFrame(columns=["full_name", "language", "pr_count", "overall_merge_rate",
                          "outsider_merge_rate", "n_outsiders"]).to_csv(OUT, index=False)

todo = pool[~pool["full_name"].isin(done_names)]
print(f"Already checked: {len(done_names)} | To check now: {len(todo)}")

for i, row in enumerate(todo.itertuples(), start=1):
    overall, outsider_rate, n_out = merge_rates(row.full_name)
    pd.DataFrame([{
        "full_name": row.full_name, "language": row.language, "pr_count": row.pr_count,
        "overall_merge_rate": overall, "outsider_merge_rate": outsider_rate, "n_outsiders": n_out,
    }]).to_csv(OUT, mode="a", header=False, index=False)
    if i % 25 == 0:
        print(f"Checked {i}/{len(todo)}")

# Summary
result = pd.read_csv(OUT).drop_duplicates("full_name", keep="last")
eligible = result[(result["overall_merge_rate"] > 0.20) & (result["outsider_merge_rate"] > 0.10)]
print("\nPassing ALL rules, by language:")
print(eligible.groupby("language").size())
print("Total eligible:", len(eligible))
print("Eligible but with fewer than 10 outsider PRs in sample:", (eligible["n_outsiders"] < 10).sum())














