import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def get_with_retry(url, params=None, retries=3):
    """One reusable GET with retries. Returns the response, or None if all attempts fail."""
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  attempt {attempt} failed ({type(e).__name__})")
            time.sleep(5 * attempt)
    return None

GRAPHQL_URL = "https://api.github.com/graphql"

def graphql(query, variables, retries=5):
    """Send a GraphQL query with retries. Waits longer when GitHub says we're going too fast."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(GRAPHQL_URL, headers=HEADERS,
                              json={"query": query, "variables": variables}, timeout=60)
            if r.status_code == 200:
                body = r.json()
                if "errors" not in body:
                    return body["data"]
                print(f"  attempt {attempt}: GraphQL errors: {body['errors']}")
                wait = 5 * attempt
            elif r.status_code in (403, 429):                         # secondary rate limit
                wait = int(r.headers.get("retry-after", 60))         # GitHub tells us how long to wait
                print(f"  attempt {attempt}: rate limited ({r.status_code}), waiting {wait}s")
            else:                                                     # e.g. 502 = server timed out
                wait = 10 * attempt
                print(f"  attempt {attempt}: status {r.status_code}, waiting {wait}s")
        except requests.exceptions.RequestException as e:
            wait = 10 * attempt
            print(f"  attempt {attempt} failed ({type(e).__name__}), waiting {wait}s")
        time.sleep(wait)
    return None