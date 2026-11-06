import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
ACTIVE_SINCE = "2026-08-05"      # Rule 2: a PR merged on or after this date

def get_with_retry(url, params=None, retries=3):
    """One reusable GET with retries. Returns the response, or None if all attempts fail."""
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  attempt {attempt} failed ({type(e).__name__})")
            time.sleep(5 * attempt)
    return None

def has_contributing(full_name):
    """Rule 4: True if the repo has a CONTRIBUTING file."""
    r = get_with_retry(f"https://api.github.com/repos/{full_name}/community/profile")
    if r is None or r.status_code != 200:
        return None
    return r.json()["files"]["contributing"] is not None

def is_active(full_name):
    """Rule 2: True if any PR was merged on or after ACTIVE_SINCE."""
    params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}
    r = get_with_retry(f"https://api.github.com/repos/{full_name}/pulls", params)
    if r is None or r.status_code != 200:
        return None
    return any(pr["merged_at"] and pr["merged_at"][:10] >= ACTIVE_SINCE for pr in r.json())


OUT = "data/candidates_rules_2_4.csv"

# Start from the Rule 3 survivors
checked = pd.read_csv("data/candidates_checked.csv").drop_duplicates("full_name", keep="last")
survivors = checked[(checked["pr_count"] >= 1000) & (checked["pr_count"] <= 20000)]

# Resume: skip repos already checked successfully
if os.path.exists(OUT):
    done = pd.read_csv(OUT)
    done_names = set(done.dropna(subset=["contributing", "active"])["full_name"])
else:
    done_names = set()
    pd.DataFrame(columns=["full_name", "language", "pr_count", "contributing", "active"]) \
      .to_csv(OUT, index=False)

todo = survivors[~survivors["full_name"].isin(done_names)]
print(f"Already checked: {len(done_names)} | To check now: {len(todo)}")

for i, row in enumerate(todo.itertuples(), start=1):
    contributing = has_contributing(row.full_name)
    active = is_active(row.full_name)
    pd.DataFrame([{
        "full_name": row.full_name, "language": row.language, "pr_count": row.pr_count,
        "contributing": contributing, "active": active,
    }]).to_csv(OUT, mode="a", header=False, index=False)
    if i % 25 == 0:
        print(f"Checked {i}/{len(todo)}")

# Summary
result = pd.read_csv(OUT).drop_duplicates("full_name", keep="last")
shortlist = result[(result["contributing"] == True) & (result["active"] == True)]
print("\nPassing Rules 1-4, by language:")
print(shortlist.groupby("language").size())
print("Total:", len(shortlist))
print("Could not check:", result[["contributing", "active"]].isna().any(axis=1).sum())