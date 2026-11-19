"""Fallback for repos where the community profile found no CONTRIBUTING file."""
import base64
from pathlib import Path
from github_client import get_with_retry

OUT = Path("data/guidelines")
MISSING = ["decentraland/builder", "nextcloud/notes", "0xMiden/protocol"]
PATHS = ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"]

for repo in MISSING:
    owner = repo.split("/")[0]
    # first the repo's own usual locations, then the organisation's shared .github repo
    candidates = [(repo, p) for p in PATHS] + [(f"{owner}/.github", p) for p in PATHS]

    for source, path in candidates:
        r = get_with_retry(f"https://api.github.com/repos/{source}/contents/{path}")
        if r is not None and r.status_code == 200:
            text = base64.b64decode(r.json()["content"]).decode("utf-8", errors="replace")
            (OUT / f"{repo.replace('/', '__')}__contributing.md").write_text(text, encoding="utf-8")
            print(f"{repo}: found {source}/{path} ({len(text)} chars)")
            break
    else:
        print(f"{repo}: not found in the repo or in {owner}/.github")