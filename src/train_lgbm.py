import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from model_config import FEATURES, TARGET, FOLDS, load_data, fold_split

BASELINE = {"roc_auc": 0.743, "pr_auc": 0.767}       # logistic regression, Day 5

df = load_data()
train = df[df["split"] == "train"]

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)

rows = []
for i, (train_end, val_start, val_end) in enumerate(FOLDS, start=1):
    tr, va = fold_split(train, train_end, val_start, val_end)
    X_tr = tr[FEATURES].astype("float64")          # plain numeric table, nothing exotic
    X_va = va[FEATURES].astype("float64")
    model = lgb.LGBMClassifier(**params).fit(
        X_tr, tr[TARGET].to_numpy(dtype="int32"),
        categorical_feature=["language_group"])    # tell LightGBM this column is a category
    score = model.predict_proba(X_va)[:, 1]
    rows.append({"fold": i,
                 "val_merge_rate": round(va[TARGET].mean(), 3),
                 "lgbm_roc_auc": round(roc_auc_score(va[TARGET], score), 3),
                 "lgbm_pr_auc": round(average_precision_score(va[TARGET], score), 3)})

results = pd.DataFrame(rows)
print(results.to_string(index=False))

mean_roc, mean_pr = results["lgbm_roc_auc"].mean(), results["lgbm_pr_auc"].mean()
print(f"\nMean ROC-AUC: {mean_roc:.3f}  (baseline {BASELINE['roc_auc']}, change {mean_roc - BASELINE['roc_auc']:+.3f})")
print(f"Mean PR-AUC:  {mean_pr:.3f}  (baseline {BASELINE['pr_auc']}, change {mean_pr - BASELINE['pr_auc']:+.3f})")