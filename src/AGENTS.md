# src/AGENTS.md — Source Directory Contract

## Purpose
This directory contains the source code for the AgroSSM pilot project. All code must comply with the invariants (INV-D1 to INV-D6, INV-A1 to INV-A6, INV-T1 to INV-T4, INV-TERM1 to INV-TERM3) and terminology constraints.

## Subdirectories
- [datacube/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/datacube) — Multi-resolution data extraction, PatchDataset, spatial blocked cross-validation split generator, and EO harmonization.
- [model/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/model) — The neural network model implementation: unified wavelength tokenizer, SelectiveScanSSM block, SlidingWindowAttention, MultiHeadLatentAttention, temporal backbone, Mixture of Experts (MoE) block, output task heads, and the top-level AgroSSM model wrapper.
- [physics/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/physics) — Mathematical physics-as-state regularizer losses (WaterBalanceStateRegularizer, AGDDStateRegularizer, StressAsymmetryRegularizer).
- [ssl/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/ssl) — Self-supervised pretext tasks (SSL-A reconstruction, SSL-B phenology, SSL-C stress trajectory, collapse monitor).
- [diagnostics/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/diagnostics) — Statistical bootstrap, causal DAG sign consistency checking, and predictive uncertainty metrics.
- [baselines/](file:///c:/Users/KIIT0001/Downloads/AgroFM/src/baselines) — Climatology, persistence, Growing Degree Days phenology model, random forest/XGBoost, and LSTM/TCN baselines.
