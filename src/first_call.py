import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept":"application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

url = "https://api.github.com/repos/fastapi/fastapi"
response = requests.get(url, headers=HEADERS, timeout=30)

print("Status code:", response.status_code)

# Body - JSON converted to a Python dict

repo = response.json()
print("Repo:",repo["full_name"])
print("Language:",repo["language"])
print("Stars:",repo["stargazers_count"])
print("Open issues + PRs",repo["open_issues_count"])

print("Rate limit:", response.headers["x-ratelimit-limit"])
print("Remaining:",response.headers["x-ratelimit-remaining"])