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