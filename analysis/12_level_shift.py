import sys
sys.path.insert(0, "src")

from model_config import FOLDS, TARGET, BEST_LGBM_PARAMS, load_data, fold_split
from train_cv import fit_predict_lgbm
from calibration import fit_level_shift

train = load_data().query("split == 'train'")

for i, (train_end, val_start, val_end) in enumerate(FOLDS, start=1):
    tr, va = fold_split(train, train_end, val_start, val_end)
    p = fit_predict_lgbm(tr, va, BEST_LGBM_PARAMS)
    actual = va[TARGET].mean()
    shift = fit_level_shift(p, actual)
    print(f"fold {i} ({val_start} to {val_end}): train rate {tr[TARGET].mean():.3f} | "
          f"predicted {p.mean():.3f} | actual {actual:.3f} | shift {shift:+.3f}")