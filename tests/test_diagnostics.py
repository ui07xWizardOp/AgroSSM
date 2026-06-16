import pytest
import numpy as np
import pandas as pd
from src.diagnostics.uncertainty import compute_ece, compute_crps, compute_picp
from src.diagnostics.bootstrap import spatial_block_bootstrap, year_level_loo_bootstrap

def test_ece():
    y_true = np.array([0, 0, 1, 1])
    # Perfect calibration
    y_prob_perf = np.array([0.1, 0.1, 0.9, 0.9])
    ece_perf = compute_ece(y_true, y_prob_perf, n_bins=2)
    assert ece_perf < 0.2

    # Bad calibration
    y_prob_bad = np.array([0.9, 0.9, 0.1, 0.1])
    ece_bad = compute_ece(y_true, y_prob_bad, n_bins=2)
    assert ece_bad > 0.8

def test_crps():
    y_true = np.array([0.0, 1.0])
    y_mean = np.array([0.0, 1.0])
    y_std = np.array([0.1, 0.1])
    crps = compute_crps(y_true, y_mean, y_std)
    assert crps > 0
    assert crps < 0.1

def test_picp():
    y_true = np.array([0.0, 1.0, 2.0])
    y_lower = np.array([-0.1, 0.9, 1.9])
    y_upper = np.array([0.1, 1.1, 2.1])
    picp = compute_picp(y_true, y_lower, y_upper)
    assert picp == 1.0

def test_spatial_bootstrap():
    df = pd.DataFrame({
        'block_id': ['b1', 'b2', 'b3', 'b4', 'b5'] * 20,
        'y_true': np.random.randn(100),
        'y_pred': np.random.randn(100)
    })

    def dummy_metric(y_t, y_p):
        return np.mean((y_t - y_p)**2)

    lower, upper = spatial_block_bootstrap(df, dummy_metric, n_resamples=10)
    assert lower <= upper

def test_loo_bootstrap():
    df = pd.DataFrame({
        'year': [2017, 2018, 2019, 2020, 2021] * 20,
        'y_true': np.random.randn(100),
        'y_pred': np.random.randn(100)
    })

    def dummy_metric(y_t, y_p):
        return np.mean((y_t - y_p)**2)

    lower, upper = year_level_loo_bootstrap(df, dummy_metric)
    assert lower <= upper
