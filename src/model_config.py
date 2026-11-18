import numpy as np
import pandas as pd

NUMERIC = ["log_additions", "log_deletions", "log_changed_files", "log_account_age",
           "log_prior_elsewhere", "n_top_dirs", "title_len", "log_body_len",
           "repo_age_days", "repo_recent_merge_rate", "log_open_backlog", "log_prs_last_30d",
           "hour_utc", "weekday"]
BOOLEAN = ["has_prior_elsewhere", "touches_tests", "docs_only", "touches_ci", "touches_deps",
           "body_empty", "mentions_issue", "has_checklist", "conventional_title"]
CATEGORICAL = ["language_group"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL
TARGET = "merged_30d"
# Human-readable names for charts, the app and the coach
FEATURE_LABELS = {
    "repo_recent_merge_rate": "Repo's recent newcomer merge rate",
    "log_additions":          "Lines added",
    "log_deletions":          "Lines deleted",
    "log_changed_files":      "Files changed",
    "n_top_dirs":             "Top-level folders touched",
    "log_open_backlog":       "Open PR backlog",
    "log_prs_last_30d":       "PRs opened in repo (last 30 days)",
    "repo_age_days":          "Repo age",
    "log_account_age":        "Author's GitHub account age",
    "log_prior_elsewhere":    "Author's prior PRs elsewhere",
    "has_prior_elsewhere":    "Has PRs elsewhere",
    "docs_only":              "Docs-only change",
    "touches_tests":          "Touches tests",
    "touches_ci":             "Touches CI config",
    "touches_deps":           "Touches dependencies",
    "title_len":              "Title length",
    "conventional_title":     "Conventional-commit title",
    "log_body_len":           "Description length",
    "body_empty":             "Empty description",
    "mentions_issue":         "Links an issue",
    "has_checklist":          "Has a checklist",
    "weekday":                "Day of week opened",
    "hour_utc":               "Hour opened (UTC)",
    "language_group":         "Language group",
}
# Expanding-window folds with a 30-day embargo: (train_end, val_start, val_end)
FOLDS = [("2023-06-01", "2023-07-01", "2024-01-01"),
         ("2023-12-01", "2024-01-01", "2024-07-01"),
         ("2024-06-01", "2024-07-01", "2025-01-01"),
         ("2024-12-01", "2025-01-01", "2025-07-01"),
         ("2025-06-01", "2025-07-01", "2025-12-02")]


def load_data():
    """Read the features and prepare types. Returns the full DataFrame (train + gap + test)."""
    df = pd.read_parquet("data/features.parquet")
    df["log_account_age"] = np.log1p(df["account_age_days"])
    df[BOOLEAN] = df[BOOLEAN].astype(int)
    df["language_group"] = df["language_group"].astype(str).astype("category").cat.codes.astype("int32")
    return df


def fold_split(train, train_end, val_start, val_end):
    """Rows to train on and rows to validate on, for one fold."""
    tr = train[train["created_at"] < train_end]
    va = train[(train["created_at"] >= val_start) & (train["created_at"] < val_end)]
    return tr, va

# Best LightGBM settings from Optuna (40 trials, recent folds weighted double), Day 6
BEST_LGBM_PARAMS = dict(
    n_estimators=422, learning_rate=0.013931194895496962, num_leaves=31,
    min_child_samples=29, subsample=0.678191614034533, subsample_freq=1,
    colsample_bytree=0.9446780534925167, reg_lambda=7.189912375733872,
    random_state=42, verbose=-1,
)