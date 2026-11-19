import joblib
import pandas as pd
import shap
from model_config import FEATURES, FEATURE_LABELS
from calibration import apply_level_shift
ACTIONABLE = ["log_additions", "log_deletions", "log_changed_files", "n_top_dirs",
              "touches_tests", "touches_ci", "touches_deps", "mentions_issue",
              "has_checklist", "title_len", "conventional_title", "log_body_len", "body_empty"]
CONTEXT = ["repo_recent_merge_rate", "log_open_backlog", "repo_age_days", "log_prs_last_30d",
           "log_account_age", "log_prior_elsewhere", "has_prior_elsewhere",
           "language_group", "weekday", "docs_only"]
HIDDEN = ["hour_utc"]

# Every feature must be in exactly one group
assert sorted(ACTIONABLE + CONTEXT + HIDDEN) == sorted(FEATURES), "feature grouping is out of date"

MIN_PUSH = 0.05


class MergeExplainer:
    def __init__(self, model_path="models/lgbm_final.joblib"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        with open("models/level_shift.json") as f:
            self.shift = json.load(f)["shift"]
    def explain(self, row, top=3):
        X = row[FEATURES].astype("float64")
        proba = float(apply_level_shift(self.model.predict_proba(X)[:, 1], self.shift)[0])
        sv = pd.Series(self.explainer(X).values[0], index=FEATURES)
                # Don't suggest adding tests to a docs-only PR
        actionable = ACTIONABLE
        if X.iloc[0]["docs_only"] == 1:
            actionable = [f for f in ACTIONABLE if f != "touches_tests"]
        def pick(features, sign=None):
            s = sv[features]
            s = s[s.abs() >= MIN_PUSH]
            if sign == "neg":
                s = s[s < 0]
            elif sign == "pos":
                s = s[s > 0]
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:top]
            return [{"label": FEATURE_LABELS.get(f, f), "push": round(float(v), 2)} for f, v in s.items()]

        return {
            "probability": round(proba, 3),
            "improve":   pick(actionable, "neg"),   # fixable things hurting the PR
            "strengths": pick(actionable, "pos"),   # fixable things already helping
            "context":   pick(CONTEXT),             # outside the author's control, either direction
        }


if __name__ == "__main__":
    import json
    from model_config import load_data

    test = load_data().query("split == 'test'")
    ex = MergeExplainer()
    for repo, number in [("JabRef/jabref", 15777), ("flatcar/nebraska", 1593)]:
        row = test[(test["repo"] == repo) & (test["number"] == number)]
        print(f"\n{repo}#{number}")
        print(json.dumps(ex.explain(row), indent=2))