"""One call for the app: PR link -> calibrated score, SHAP explanation, coach advice."""
import csv
import json
import sys
from coach import Coach
from explain import MergeExplainer
from live_pr import fetch_pr, live_features, parse_pr_url

# The 50 projects the model covers; keys are lower-case so "jabref/jabref" also matches
with open("repos.csv") as f:
    SUPPORTED = {r["full_name"].lower(): r["full_name"] for r in csv.DictReader(f)}


def check_url(url):
    """Cheap check before any API call: a PR link to one of the 50 projects. Returns the canonical link."""
    repo, number = parse_pr_url(url)
    if repo.lower() not in SUPPORTED:
        raise ValueError(f"{repo} isn't one of the 50 projects MergeMate covers. "
                         "See the list of supported projects.")
    return f"https://github.com/{SUPPORTED[repo.lower()]}/pull/{number}"


class MergeMate:
    def __init__(self):
        self.explainer = MergeExplainer()     # loads the model, SHAP and the calibration shift once
        self.coach = Coach()                  # loads the embedding model, Chroma and Claude once

    def analyze(self, url, with_coach=True, llm_model=None):
        pr = fetch_pr(check_url(url))
        X = live_features(pr)
        result = self.explainer.explain(X)

        warnings = []
        if pr["author_prior_prs_here"]:
            warnings.append(f"The author has {pr['author_prior_prs_here']} earlier PR(s) in this repo. "
                            "The model is built for first PRs, so treat the score with care.")
        if pr["state"] != "OPEN":
            warnings.append(f"This PR is {pr['state'].lower()}. The score describes it as it was opened, "
                            "but the open-PR backlog is today's count.")

        result.update({"repo": pr["repo"], "number": pr["number"], "title": pr["title"],
                       "warnings": warnings})
        if with_coach:
            result["coach"] = self.coach.review(pr["repo"], pr, llm_model=llm_model)["advice"]
        return result


if __name__ == "__main__":
    out = MergeMate().analyze(sys.argv[1])
    advice = out.pop("coach", "")
    print(json.dumps(out, indent=2))
    print("\nCOACH:\n" + advice)