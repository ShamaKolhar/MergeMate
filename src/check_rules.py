import os
import re
import requests
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def count_prs(full_name, retries=3):
    """Return total PRs (all states) using one request. Retries on network errors."""
    url = f"https://api.github.com/repos/{full_name}/pulls"
    params = {"state": "all", "per_page": 1}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            break                                         # success, so leave the retry loop
        except requests.exceptions.RequestException as e:
            print(f"  {full_name}: attempt {attempt} failed ({type(e).__name__})")
            time.sleep(5 * attempt)                       # wait 5s, then 10s, then 15s
    else:
        return None                                       # every attempt failed

    if response.status_code != 200:
        return None

    if "last" in response.links:
        last_url = response.links["last"]["url"]
        return int(re.search(r"[?&]page=(\d+)", last_url).group(1))

    return len(response.json())


OUT = "data/candidates_checked.csv"
df = pd.read_csv("data/candidates.csv")


if os.path.exists(OUT):
    done = pd.read_csv(OUT)
    done_names = set(done.dropna(subset=["pr_count"])["full_name"])
else:
    done_names = set()
    pd.DataFrame(columns=["full_name", "language", "pr_count"]).to_csv(OUT, index=False)

todo = df[~df["full_name"].isin(done_names)]
print(f"Already checked: {len(done_names)} | To check now: {len(todo)}")

for i, row in enumerate(todo.itertuples(), start=1):
    pr_count = count_prs(row.full_name)
   
    pd.DataFrame([{"full_name": row.full_name, "language": row.language, "pr_count": pr_count}]) \
      .to_csv(OUT, mode="a", header=False, index=False)
    if i % 50 == 0:
        print(f"Checked {i}/{len(todo)}")


result = pd.read_csv(OUT).drop_duplicates(subset="full_name", keep="last")
passed = result[(result["pr_count"] >= 1000) & (result["pr_count"] <= 20000)]
print("\nPassing Rule 3, by language:")
print(passed.groupby("language").size())
print("Total passing Rule 3:", len(passed))
print("Could not check:", result["pr_count"].isna().sum())