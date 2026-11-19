"""Refit the app's level shift on the most recent 3 months of known outcomes."""
import json
import joblib
import pandas as pd
from model_config import FEATURES, TARGET, load_data
from calibration import fit_level_shift

WINDOW_DAYS = 90

df = load_data()
end = df["created_at"].max()
recent = df[df["created_at"] >= end - pd.Timedelta(days=WINDOW_DAYS)]

model = joblib.load("models/lgbm_final.joblib")
p = model.predict_proba(recent[FEATURES].astype("float64"))[:, 1]
shift = fit_level_shift(p, recent[TARGET].mean())

info = {"shift": round(float(shift), 3), "window_days": WINDOW_DAYS,
        "window_end": str(end.date()), "n_prs": int(len(recent)),
        "actual_rate": round(float(recent[TARGET].mean()), 3),
        "mean_pred_raw": round(float(p.mean()), 3)}
with open("models/level_shift.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))