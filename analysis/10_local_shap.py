import sys
sys.path.insert(0, "src")

import joblib
import numpy as np
import pandas as pd
import shap
from model_config import FEATURES, FEATURE_LABELS, load_data

df = load_data()
test = df[df["split"] == "test"].reset_index(drop=True)
X = test[FEATURES].astype("float64")

model = joblib.load("models/lgbm_final.joblib")
explainer = shap.TreeExplainer(model)
proba = model.predict_proba(X)[:, 1]


def explain(i, top=5):
    sv = explainer(X.iloc[[i]])
    contrib = pd.Series(sv.values[0], index=[FEATURE_LABELS.get(f, f) for f in FEATURES])

    name = (f"{test.loc[i, 'repo']}#{test.loc[i, 'number']}"
            if {"repo", "number"} <= set(test.columns) else f"test row {i}")
    print(f"\n{name}: predicted {proba[i]:.0%}, actually merged = {bool(test.loc[i, 'merged_30d'])}")

    # Additivity check: baseline + all contributions = the model's score (log-odds)
    total = sv.base_values[0] + contrib.sum()
    print(f"  baseline {sv.base_values[0]:+.2f} + contributions {contrib.sum():+.2f} = {total:+.2f}"
          f"  (model log-odds {np.log(proba[i] / (1 - proba[i])):+.2f})")

    for feat in contrib.abs().sort_values(ascending=False).index[:top]:
        print(f"  {contrib[feat]:+.2f}  {feat}")


order = np.argsort(proba)
explain(int(order[len(order) // 4]))        # a low-scoring PR (25th percentile)
explain(int(order[int(len(order) * 0.9)]))  # a high-scoring PR (90th percentile)
