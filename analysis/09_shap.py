import sys
sys.path.insert(0, "src")
from matplotlib.colors import LinearSegmentedColormap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from model_config import FEATURES, FEATURE_LABELS, load_data

df = load_data()
test = df[df["split"] == "test"]
X = test[FEATURES].astype("float64")

model = joblib.load("models/lgbm_final.joblib")
explainer = shap.TreeExplainer(model)
sv = explainer(X)                     # one SHAP value per PR per feature (in log-odds units)

# ---- Global importance: average size of each feature's contribution ----
importance = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print("Mean |SHAP| on the 2026 test set (higher = more influence):")
print(importance.round(3).to_string())

# ---- Beeswarm chart, in the README palette ----
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MID = "#2a78d6", "#eb6834", "#b8b6b0"

# low feature value = blue, high = orange, neutral grey in the middle
cmap = LinearSegmentedColormap.from_list("mergemate", [BLUE, MID, ORANGE])
sv.feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES]
shap.plots.beeswarm(sv, max_display=12, show=False, color=cmap, plot_size=(9, 7))

fig = plt.gcf()
fig.patch.set_facecolor(SURFACE)
for ax in fig.axes:                       # main plot + colour bar
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_color(GRID)

main = fig.axes[0]
main.set_xlabel("SHAP value (right = pushes towards merged)", color=INK_2)
main.set_title("What drives the merge prediction", loc="left",
               fontsize=13, color=INK, pad=12)
fig.text(0.01, -0.01,
         "Source: MergeMate, tuned LightGBM on 3,336 newcomer PRs opened in 2026 (test set)",
         fontsize=8, color=INK_2)

plt.savefig("docs/figures/04_shap_summary.png", dpi=150,
            bbox_inches="tight", facecolor=SURFACE)
print("\nSaved docs/figures/04_shap_summary.png")