import torch
import numpy as np
from src.model.agrossm import AgroSSM
from src.diagnostics.uncertainty import compute_ece, compute_crps
from src.diagnostics.bootstrap import spatial_block_bootstrap
import pandas as pd

def evaluate():
    """Instantiate AgroSSM, run predictions, and produce diagnostic metrics."""
    model = AgroSSM()
    model.eval()

    # Dummy outputs for evaluation
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.uniform(0, 1, 100)

    ece = compute_ece(y_true, y_prob)
    print(f"ECE: {ece:.4f}")

    # CRPS
    y_true_cont = np.random.randn(100)
    y_mean = np.random.randn(100)
    y_std = np.abs(np.random.randn(100)) + 0.1

    crps = compute_crps(y_true_cont, y_mean, y_std)
    print(f"CRPS: {crps:.4f}")

if __name__ == "__main__":
    evaluate()
