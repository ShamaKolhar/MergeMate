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

def graphql(query, variables, retries=3):
    """Send a GraphQL query with retries. Returns the 'data' part, or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(GRAPHQL_URL, headers=HEADERS,
                              json={"query": query, "variables": variables}, timeout=60)
            body = r.json()
            if r.status_code == 200 and "errors" not in body:
                return body["data"]
            print(f"  GraphQL problem (status {r.status_code}): {body.get('errors') or body}")
        except requests.exceptions.RequestException as e:
            print(f"  attempt {attempt} failed ({type(e).__name__})")
        time.sleep(5 * attempt)
    return None