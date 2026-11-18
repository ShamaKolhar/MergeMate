from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from model_config import FEATURES, TARGET, BEST_LGBM_PARAMS, load_data
from train_cv import cv_scores, fit_predict_lr, fit_predict_lgbm

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mergemate-cv")

df = load_data()
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# ---- Check 1: tuned settings on the PLAIN mean of the CV folds ----
rocs, prs = cv_scores(fit_predict_lgbm, BEST_LGBM_PARAMS, train)
print("Tuned LightGBM on CV folds")
print(f"  plain mean ROC-AUC {np.mean(rocs):.3f} | plain mean PR-AUC {np.mean(prs):.3f} | last fold ROC-AUC {rocs[-1]:.3f}")
print("  (default LightGBM: 0.746 / 0.760 / 0.730; logistic regression: 0.743 / 0.767 / 0.704)")

# ---- Check 2: the final, one-time test on 2026 ----
print(f"\n2026 test set: {len(test):,} PRs, actual merge rate {test[TARGET].mean():.3f}")
with mlflow.start_run(run_name="final_test_2026"):
    for name, fit_predict, params in [("logistic_regression", fit_predict_lr, dict(max_iter=1000)),
                                      ("lightgbm_tuned", fit_predict_lgbm, BEST_LGBM_PARAMS)]:
        score = fit_predict(train, test, params)          # trained on ALL training data
        roc = roc_auc_score(test[TARGET], score)
        pr = average_precision_score(test[TARGET], score)
        brier = brier_score_loss(test[TARGET], score)
        mlflow.log_metric(f"{name}_test_roc_auc", roc)
        mlflow.log_metric(f"{name}_test_pr_auc", pr)
        mlflow.log_metric(f"{name}_test_brier", brier)
        print(f"  {name:20} ROC-AUC {roc:.3f} | PR-AUC {pr:.3f} | Brier {brier:.3f} | "
              f"mean predicted {score.mean():.3f}")

# ---- Save the final model for the app (Day 11) ----
import lightgbm as lgb
final_model = lgb.LGBMClassifier(**BEST_LGBM_PARAMS).fit(
    train[FEATURES].astype("float64"), train[TARGET].to_numpy(dtype="int32"),
    categorical_feature=["language_group"])
Path("models").mkdir(exist_ok=True)
joblib.dump(final_model, "models/lgbm_final.joblib")
print("\nSaved models/lgbm_final.joblib")