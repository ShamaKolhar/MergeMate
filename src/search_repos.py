import os
import time
import requests
import random
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization":f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version":"2022-11-28",
}

LANGUAGES = ["python", "javascript", "typescript", "go", "rust", "java"]
SEARCH_URL = "https://api.github.com/search/repositories"

candidates =[]

random.seed(42)          # makes the random sample reproducible
SAMPLE_SIZE = 100        # candidates per language
candidates = []

for lang in LANGUAGES:
    query = f"language:{lang} good-first-issues:>5 archived:false fork:false pushed:>2026-08-05"
    pool = []            # every repo that matches, for this language
    page = 1

    while page <= 10:    # GitHub caps search results at 1,000 (10 pages x 100)
        params = {"q": query, "per_page": 100, "page": page}
        response = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=30)

        if response.status_code != 200:          # stop cleanly if something goes wrong
            print(f"  Error on {lang} page {page}: {response.status_code}")
            break

        items = response.json()["items"]
        if not items:                            # empty page means the pool is finished
            break

        pool.extend(repo["full_name"] for repo in items)
        page += 1
        time.sleep(3)                            # stay under 30 searches per minute

    # take a random sample from the pool
    sample = random.sample(pool, min(SAMPLE_SIZE, len(pool)))
    print(f"{lang:12} pool fetched: {len(pool):5} | sampled: {len(sample)}")

    for name in sample:
        candidates.append({"full_name": name, "language": lang})

print("\nTotal candidates:", len(candidates))
print("Random 5:", [c["full_name"] for c in candidates[:5]])

df=pd.DataFrame(candidates)
df.to_csv("data/candidates.csv",index=False)
print("Saved to candidates.csv")