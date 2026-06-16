import numpy as np
import pandas as pd
from typing import Callable, Tuple

def spatial_block_bootstrap(df: pd.DataFrame, metric_fn: Callable, n_resamples: int = 1000) -> Tuple[float, float]:
    """
    Spatial Block Bootstrap (50km blocks).
    Returns 95% CI for the given metric.
    """
    blocks = df['block_id'].unique()
    metrics = []

    for _ in range(n_resamples):
        sampled_blocks = np.random.choice(blocks, size=len(blocks), replace=True)
        # Create bootstrap sample
        resampled_dfs = [df[df['block_id'] == b] for b in sampled_blocks]
        resampled_df = pd.concat(resampled_dfs)

        m = metric_fn(resampled_df['y_true'].values, resampled_df['y_pred'].values)
        metrics.append(m)

    return np.percentile(metrics, 2.5), np.percentile(metrics, 97.5)

def year_level_loo_bootstrap(df: pd.DataFrame, metric_fn: Callable) -> Tuple[float, float]:
    """
    Year-Level Leave-One-Out Bootstrap.
    """
    years = df['year'].unique()
    metrics = []

    for y in years:
        loo_df = df[df['year'] != y]
        m = metric_fn(loo_df['y_true'].values, loo_df['y_pred'].values)
        metrics.append(m)

    return np.percentile(metrics, 2.5), np.percentile(metrics, 97.5)
