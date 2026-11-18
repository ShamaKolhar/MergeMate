import mlflow
import numpy as np
import optuna
from model_config import load_data
from train_cv import cv_scores, fit_predict_lgbm

FOLD_WEIGHTS = [1, 1, 1, 2, 2]
N_TRIALS = 40

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mergemate-cv")
train = load_data().query("split == 'train'")


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=42, verbose=-1,
    )
    rocs, prs = cv_scores(fit_predict_lgbm, params, train)
    weighted = np.average(rocs, weights=FOLD_WEIGHTS)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):   # each trial = one child run
        mlflow.log_params(params)
        mlflow.log_metric("weighted_roc_auc", weighted)
        mlflow.log_metric("mean_roc_auc", np.mean(rocs))
        mlflow.log_metric("mean_pr_auc", np.mean(prs))
    return weighted


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)                   # quieter output
    with mlflow.start_run(run_name="lightgbm_optuna"):                     # parent run for the whole search
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best = study.best_trial
        mlflow.log_params(best.params)
        mlflow.log_metric("best_weighted_roc_auc", best.value)

    print("\nBest weighted ROC-AUC:", round(best.value, 4))
    print("Best parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")