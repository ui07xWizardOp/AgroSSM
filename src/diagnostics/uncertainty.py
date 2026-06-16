import numpy as np
from typing import Tuple

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for binary classification."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            true_mean = np.mean(y_true[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(prob_mean - true_mean)

    return float(ece)

def compute_crps(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> float:
    """Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions."""
    from scipy.stats import norm

    # CRPS for N(mu, sigma^2)
    # sigma * (z * (2Phi(z) - 1) + 2phi(z) - 1/sqrt(pi)) where z = (y - mu)/sigma
    y_pred_std = np.clip(y_pred_std, 1e-6, None)
    z = (y_true - y_pred_mean) / y_pred_std

    phi = norm.pdf(z)
    Phi = norm.cdf(z)

    crps = y_pred_std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))

def compute_picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP)."""
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))
