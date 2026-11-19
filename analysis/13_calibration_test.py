import sys
sys.path.insert(0, "src")

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from model_config import FEATURES, TARGET, load_data
from calibration import apply_level_shift

SHIFT = -0.160   # mean of CV folds 3-5 (the drift era), chosen before looking at 2026

test = load_data().query("split == 'test'")
X = test[FEATURES].astype("float64")
y = test[TARGET].astype(int).to_numpy()
p_raw = joblib.load("models/lgbm_final.joblib").predict_proba(X)[:, 1]
p_shifted = apply_level_shift(p_raw, SHIFT)


def calibration_table(p, y):
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    table = (pd.DataFrame({"bin": bins, "pred": p, "actual": y})
               .groupby("bin")
               .agg(n=("pred", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean")))
    table["gap"] = table["mean_pred"] - table["actual_rate"]
    ece = (table["n"] * table["gap"].abs()).sum() / table["n"].sum()
    return table, ece


for name, p in [("raw", p_raw), ("shifted", p_shifted)]:
    table, ece = calibration_table(p, y)
    print(f"\n{name.upper()}: mean predicted {p.mean():.3f} vs actual {y.mean():.3f} | "
          f"ECE {ece:.3f} | Brier {brier_score_loss(y, p):.3f} | ROC-AUC {roc_auc_score(y, p):.3f}")
    print(table.round(3).to_string())