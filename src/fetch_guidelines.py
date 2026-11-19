"""Download each repo's CONTRIBUTING file and PR template from GitHub."""
import base64
import csv
from pathlib import Path
from github_client import get_with_retry

OUT = Path("data/guidelines")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_file(api_url):
    """Fetch one file through the contents API and decode it to text."""
    r = get_with_retry(api_url)
    if r is None or r.status_code != 200:
        return None
    return base64.b64decode(r.json()["content"]).decode("utf-8", errors="replace")


with open("repos.csv") as f:
    repos = [row["full_name"] for row in csv.DictReader(f)]

summary = []
for i, repo in enumerate(repos, start=1):
    r = get_with_retry(f"https://api.github.com/repos/{repo}/community/profile")
    files = r.json().get("files", {}) if r is not None and r.status_code == 200 else {}

    found = {}
    for kind in ["contributing", "pull_request_template"]:
        info = files.get(kind)
        text = fetch_file(info["url"]) if info else None
        if text:
            (OUT / f"{repo.replace('/', '__')}__{kind}.md").write_text(text, encoding="utf-8")
        found[kind] = len(text) if text else 0

    summary.append((repo, found["contributing"], found["pull_request_template"]))
    print(f"[{i}/{len(repos)}] {repo}: contributing {found['contributing']} chars, "
          f"PR template {found['pull_request_template']} chars")

n_c = sum(1 for _, c, _ in summary if c)
n_t = sum(1 for _, _, t in summary if t)
print(f"\nCONTRIBUTING found for {n_c}/50 repos, PR template for {n_t}/50")