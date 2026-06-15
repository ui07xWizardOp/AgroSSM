# tests/AGENTS.md — Test Directory

## Purpose
This directory contains the unit tests and integration tests for all components of the AgroSSM model and dataset. All tests must be run using PyTest: `pytest tests/`

## Test Files
- `test_datacube.py` — Verifies PatchDataset schema, shapes, normalization, split generator leakage, and channel registry.
- `test_model.py` — Verifies unified tokenizer mapping, SelectiveScanSSM parameters isolation, SlidingWindowAttention, MultiHeadLatentAttention, MoE routing balance, output heads shapes, and top-level forward passes.
- `test_physics.py` — Verifies WaterBalanceStateRegularizer, AGDDStateRegularizer, StressAsymmetryRegularizer, and Softplus activations.
- `test_ssl.py` — Verifies the pretext task loaders and loss computation.
- `test_diagnostics.py` — Verifies causal DAG, bootstrap, and uncertainty metrics calculations.
