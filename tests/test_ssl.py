import pytest
import torch
from src.ssl.ssl_a_reconstruction import MaskedReconstructionLoss
from src.ssl.ssl_b_phenology import PhenologyTransitionLoss
from src.ssl.ssl_c_trajectory import DroughtTrajectoryLoss
from src.ssl.collapse_monitor import EmbeddingCollapseMonitor

def test_reconstruction_loss():
    loss_fn = MaskedReconstructionLoss()
    pred = torch.randn(2, 10, 256)
    target = torch.randn(2, 10, 256)
    mask = torch.randint(0, 2, (2, 10))
    loss = loss_fn(pred, target, mask)
    assert loss.item() >= 0

def test_phenology_loss():
    loss_fn = PhenologyTransitionLoss()
    pred = torch.randn(2, 30, 1)
    target = torch.randn(2, 30)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0

def test_trajectory_loss():
    loss_fn = DroughtTrajectoryLoss()
    pred = torch.randn(2, 30, 3)
    target = torch.randn(2, 30, 3)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0

def test_collapse_monitor():
    monitor = EmbeddingCollapseMonitor()

    # Random embeddings should not be collapsed
    good_embeddings = torch.randn(2, 10, 64)
    ratio_good = monitor(good_embeddings)
    assert ratio_good < 0.90

    # Collapsed embeddings
    collapsed_embeddings = torch.ones(2, 10, 64)
    ratio_bad = monitor(collapsed_embeddings)
    # Should be 1.0 (or very close) indicating complete collapse
    assert ratio_bad >= 0.90
