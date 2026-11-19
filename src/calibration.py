"""Level correction for drift: shift every score's log-odds by one constant."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit


def fit_level_shift(p, actual_rate):
    """The constant that makes the average shifted prediction equal actual_rate."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return brentq(lambda d: expit(z + d).mean() - actual_rate, -5, 5)


def apply_level_shift(p, shift):
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + shift)