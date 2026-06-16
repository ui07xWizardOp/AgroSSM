import pytest
import torch
from src.physics.state_regularizer import SharedPhysicsDecoders, WaterBalanceStateRegularizer, AGDDStateRegularizer, StressAsymmetryRegularizer

def test_water_balance():
    decoders = SharedPhysicsDecoders(64)
    reg = WaterBalanceStateRegularizer(decoders.decode_SM)

    h_water = torch.randn(2, 30, 64)
    precip = torch.zeros(2, 30)
    et = torch.zeros(2, 30)
    i_obs = torch.zeros(2, 30)
    i_latent = torch.zeros(2, 30)
    soil = torch.zeros(2, 7)

    loss = reg(h_water, precip, et, i_obs, i_latent, soil)
    assert loss >= 0

def test_agdd():
    decoders = SharedPhysicsDecoders(64)
    reg = AGDDStateRegularizer(decoders.decode_AGDD)

    h_agdd = torch.randn(2, 30, 64)
    tmean = torch.full((2, 30), 20.0)
    planting = torch.zeros(2)
    doy = torch.arange(30).unsqueeze(0).repeat(2, 1)

    loss = reg(h_agdd, tmean, planting, doy)
    assert loss >= 0

def test_stress_asymmetry():
    decoders = SharedPhysicsDecoders(64)
    reg = StressAsymmetryRegularizer(decoders.decode_stress)

    h_stress = torch.randn(2, 30, 64)
    loss = reg(h_stress)
    assert loss >= 0
