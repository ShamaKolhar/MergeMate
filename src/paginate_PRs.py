import os
import requests
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization":f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-Github-Api-Version":"2022-11-28",
}

url="https://api.github.com/repos/fastapi/fastapi/pulls"
all_prs=[]
page=1
MAX_PAGES=3

while page<= MAX_PAGES:
    params = {"state":"closed","per_page":100,"page":page}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    batch = response.json()

    if not batch:
        break

    all_prs.extend(batch) # adds each page's PR to one big list
    print(f"Page {page}: got {len(batch)} PRs | Request left: {response.headers['x-ratelimit-remaining']}")
    page+=1

print("\nTotal PRs collected: ", len(all_prs))

merged = sum(1 for pr in all_prs if pr["merged_at"] is not None)  #counts merged PRs
newcomers = sum(1 for pr in all_prs if pr["author_association"] == "FIRST_TIME_CONTRIBUTOR") #finds newcomer PRs
print("Merged:", merged)
print("From first-time contributors:", newcomers)


""" TESTING THEORY: Is author_assocation leaky ? 
GitHub calculates author_association as of today, not as of when the PR was opened. SO, newcomes whose PR got merged now shows as CONTRIBUTOR and the rejected ones show as NONE, so the field reveals the outcome, resulting in a TARGET LEAKAGE.
TEST : Compare merge rates across association groups
RESULTS : None 0%, CONTRIBUTOR 94%, MEMBER 98%, which confirms the field is leaky.
DECISION: Not to use author_association as a feature"""

from collections import Counter

print("\nAuthor association counts:")
print(Counter(pr["author_association"] for pr in all_prs))

print("\nMerge rate by association:")
for assoc in set(pr["author_association"] for pr in all_prs):
    group = [pr for pr in all_prs if pr["author_association"] == assoc]
    merged_in_group = sum(1 for pr in group if pr["merged_at"] is not None)
    print(f"{assoc:15} PRs: {len(group):4}  merged: {merged_in_group / len(group):.0%}")