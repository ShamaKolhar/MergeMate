import sys
sys.path.insert(0, "src")

import joblib
import pandas as pd
from model_config import FEATURES, TARGET, load_data

test = load_data().query("split == 'test'")
X = test[FEATURES].astype("float64")
y = test[TARGET].astype(int).to_numpy()
p = joblib.load("models/lgbm_final.joblib").predict_proba(X)[:, 1]

# 10 equal-sized groups, from lowest to highest predicted probability
bins = pd.qcut(p, 10, labels=False, duplicates="drop")
table = (pd.DataFrame({"bin": bins, "pred": p, "actual": y})
           .groupby("bin")
           .agg(n=("pred", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean")))
table["gap"] = table["mean_pred"] - table["actual_rate"]   # positive = overconfident
print(table.round(3).to_string())

ece = (table["n"] * table["gap"].abs()).sum() / table["n"].sum()
print(f"\nOverall: mean predicted {p.mean():.3f} vs actual {y.mean():.3f}")
print(f"Expected calibration error (ECE): {ece:.3f}")