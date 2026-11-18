import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from model_config import NUMERIC, BOOLEAN, CATEGORICAL, FEATURES, TARGET, FOLDS, load_data, fold_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")      # results stored in a local database file
mlflow.set_experiment("mergemate-cv")


def fit_predict_lr(tr, va, params):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC + BOOLEAN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(**params))])
    model.fit(tr[FEATURES], tr[TARGET])
    return model.predict_proba(va[FEATURES])[:, 1]


def fit_predict_lgbm(tr, va, params):
    model = lgb.LGBMClassifier(**params).fit(
        tr[FEATURES].astype("float64"), tr[TARGET].to_numpy(dtype="int32"),
        categorical_feature=["language_group"])
    return model.predict_proba(va[FEATURES].astype("float64"))[:, 1]

def cv_scores(fit_predict, params, train):
    """ROC-AUC and PR-AUC for each fold (no logging)."""
    rocs, prs = [], []
    for train_end, val_start, val_end in FOLDS:
        tr, va = fold_split(train, train_end, val_start, val_end)
        score = fit_predict(tr, va, params)
        rocs.append(roc_auc_score(va[TARGET], score))
        prs.append(average_precision_score(va[TARGET], score))
    return rocs, prs

def run_cv(name, fit_predict, params, train):
    """Run all folds for one model and log everything to MLflow. Returns mean ROC-AUC and PR-AUC."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        rocs, prs = [], []
        for i, (train_end, val_start, val_end) in enumerate(FOLDS, start=1):
            tr, va = fold_split(train, train_end, val_start, val_end)
            score = fit_predict(tr, va, params)
            roc = roc_auc_score(va[TARGET], score)
            pr = average_precision_score(va[TARGET], score)
            mlflow.log_metric("roc_auc", roc, step=i)          # one point per fold
            mlflow.log_metric("pr_auc", pr, step=i)
            rocs.append(roc); prs.append(pr)
        mlflow.log_metric("mean_roc_auc", np.mean(rocs))
        mlflow.log_metric("mean_pr_auc", np.mean(prs))
        mlflow.log_metric("last_fold_roc_auc", rocs[-1])      # the fold most like 2026
    print(f"{name:20} mean ROC-AUC {np.mean(rocs):.3f} | mean PR-AUC {np.mean(prs):.3f} | last fold ROC-AUC {rocs[-1]:.3f}")


if __name__ == "__main__":
    train = load_data().query("split == 'train'")

    run_cv("logistic_regression", fit_predict_lr, dict(max_iter=1000), train)
    run_cv("lightgbm_default", fit_predict_lgbm,
           dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=42, verbose=-1), train)