import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_parquet("data/features.parquet")
df["log_account_age"] = np.log1p(df["account_age_days"])
train = df[df["split"] == "train"].copy()

NUMERIC = ["log_additions", "log_deletions", "log_changed_files", "log_account_age",
           "log_prior_elsewhere", "n_top_dirs", "title_len", "log_body_len",
           "repo_age_days", "repo_recent_merge_rate", "log_open_backlog", "log_prs_last_30d",
           "hour_utc", "weekday"]
BOOLEAN = ["has_prior_elsewhere", "touches_tests", "docs_only", "touches_ci", "touches_deps",
           "body_empty", "mentions_issue", "has_checklist", "conventional_title"]
CATEGORICAL = ["language_group"]
TARGET = "merged_30d"

# Expanding-window folds: (train only on PRs opened before 'train_end', validate on [val_start, val_end))
FOLDS = [("2023-06-01", "2023-07-01", "2024-01-01"),
         ("2023-12-01", "2024-01-01", "2024-07-01"),
         ("2024-06-01", "2024-07-01", "2025-01-01"),
         ("2024-12-01", "2025-01-01", "2025-07-01"),
         ("2025-06-01", "2025-07-01", "2025-12-02")]

def make_model():
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC + BOOLEAN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])

rows = []
for i, (train_end, val_start, val_end) in enumerate(FOLDS, start=1):
    tr = train[train["created_at"] < train_end]
    va = train[(train["created_at"] >= val_start) & (train["created_at"] < val_end)]

    # Level 1: simple rule = the repo's recent merge rate (missing filled with the training average)
    rule_score = va["repo_recent_merge_rate"].fillna(tr["repo_recent_merge_rate"].mean())

    # Level 2: logistic regression on all features
    model = make_model().fit(tr[NUMERIC + BOOLEAN + CATEGORICAL], tr[TARGET])
    lr_score = model.predict_proba(va[NUMERIC + BOOLEAN + CATEGORICAL])[:, 1]

    rows.append({
        "fold": i, "train_rows": len(tr), "val_rows": len(va),
        "val_merge_rate": round(va[TARGET].mean(), 3),
        "rule_roc_auc": round(roc_auc_score(va[TARGET], rule_score), 3),
        "lr_roc_auc":   round(roc_auc_score(va[TARGET], lr_score), 3),
        "rule_pr_auc":  round(average_precision_score(va[TARGET], rule_score), 3),
        "lr_pr_auc":    round(average_precision_score(va[TARGET], lr_score), 3),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
print("\nMean across folds:")
print(results[["rule_roc_auc", "lr_roc_auc", "rule_pr_auc", "lr_pr_auc"]].mean().round(3).to_string())