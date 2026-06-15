# AGENTS.md — AgroEarthFM / AgroSSM Root

## Project Identity

- **Name:** AgroEarthFM (codename: AgroSSM for the v3.0 compute-efficient architecture)
- **Type:** Research prototype — machine learning model for agro-ecosystem prediction
- **Status:** Pre-implementation. All documentation is complete. No source code exists yet.
- **Active Branch:** `architecture-v3-agrossm`
- **Owner:** Priyobrata Chatterjee (unknowninitialiser07@gmail.com)
- **Repository:** https://gitlab.com/personal-group8591224/Personal-project.git

---

## What This Project Is

AgroSSM is a from-scratch, compute-efficient, process-structured agro-ecosystem AI model. It fuses satellite imagery, weather data, soil maps, and crop calendars through a single unified tokenizer, processes them with a linear-time hybrid state-space backbone whose hidden state encodes physical accumulators (water balance, heat accumulation, crop stress), and routes capacity through a regime-specialized Mixture-of-Experts. It predicts crop stress, phenology transitions, and yield-risk anomalies.

**The central thesis:** Process structure can replace scale — a small model with built-in agronomic physics can match large brute-force models at a fraction of the compute.

---

## Document Hierarchy (Which Document Controls What)

| Document | Role | Status |
|----------|------|--------|
| `Documentation/AgroEarthFM_MVP_Spec_v2.3.md` | **FROZEN master specification.** Controls: claims (C1–C6), data licensing, no-leakage protocol, evaluation metrics, terminology constraints, success criteria, baselines, risk register, and the full 52-week roadmap. | Frozen. Do not modify. |
| `Documentation/AgroEarthFM_Architecture_v3_AgroSSM.md` | **Active architecture specification.** Controls: model architecture (unified tokenizer, hybrid backbone, MoE, physics-as-state-regularizer), parameter budgets, compute projections, and pilot scope. Supersedes v2.3 architecture only. | Active. Update when architecture decisions change. |
| `Documentation/AgroSSM_Implementation_Plan.md` | **Layman-friendly implementation guide.** Covers everything from problem context to sprint-by-sprint roadmap with acceptance criteria. Read this first to understand the project. | Active. Update as implementation progresses. |
| `Documentation/AgroSSM_Agent_Handoff.md` | **Agent execution instructions.** Step-by-step coding instructions for each module, with exact file paths, function signatures, tensor shapes, test commands, and failure handling. This is the primary instruction document for any AI agent performing implementation. | Active. Update as code is written. |

---

## Directory Structure (Current → Target)

```
AgroFM/                              ← you are here (repository root)
├── AGENTS.md                        ← THIS FILE (root contract)
├── Documentation/                   ← Research specifications and plans
│   ├── AGENTS.md                    ← Documentation subdirectory contract
│   └── ...                          ← All spec documents (see Documentation/AGENTS.md)
├── config/                          ← TO BE CREATED: YAML configuration files
│   ├── AGENTS.md
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── ssl_config.yaml
├── src/                             ← TO BE CREATED: All source code
│   ├── AGENTS.md
│   ├── datacube/                    ← Data loading and patch extraction
│   ├── model/                       ← AgroSSM model architecture
│   │   ├── tokenizer.py             ← Unified wavelength-conditioned tokenizer
│   │   ├── ssm_block.py             ← State-space model block (physical accumulator)
│   │   ├── swa_block.py             ← Sliding-window attention block
│   │   ├── mla_block.py             ← Multi-head latent attention block
│   │   ├── backbone.py              ← Hybrid temporal backbone (interleaves blocks)
│   │   ├── moe.py                   ← Regime-routed Mixture-of-Experts
│   │   ├── heads.py                 ← Primary + auxiliary output heads
│   │   └── agrossm.py               ← Top-level model class
│   ├── physics/                     ← Physics-as-state-regularizer losses
│   ├── ssl/                         ← Self-supervised pretext tasks
│   ├── diagnostics/                 ← Causal DAG, bootstrap, uncertainty
│   └── baselines/                   ← Baseline models
├── scripts/                         ← TO BE CREATED: CLI entrypoints
├── tests/                           ← TO BE CREATED: PyTest test suite
└── .gitignore
```

---

## Global Rules (Apply to ALL Subdirectories)

1. **No code without a spec.** Every module must trace back to a section in the v2.3 spec or the v3.0 architecture doc.
2. **No forced 10m upsampling.** Coarse data (weather at 31km, soil at 250m) stays at native resolution. The model handles multi-resolution via the MLA cross-modal block.
3. **Physics on the state, not just the output.** Water-balance and AGDD constraints are applied to the SSM hidden sub-states (`h_water`, `h_agdd`), not only to the output predictions.
4. **Tests before merge.** Every module must have unit tests that pass before the module is considered complete.
5. **Checkpoint aggressively.** Colab sessions disconnect. Save model state every epoch.
6. **Terminology compliance.** Never use prohibited terms (see v2.3 Section 22). Use "I_latent" not "irrigation detection". Use "structural prior" not "causal discovery".

---

## Child AGENTS.md Index

| Path | Scope |
|------|-------|
| `Documentation/AGENTS.md` | Research specification documents |
| `src/AGENTS.md` | All source code (to be created with code) |
| `config/AGENTS.md` | Configuration files (to be created with configs) |
| `tests/AGENTS.md` | Test suite (to be created with tests) |
| `scripts/AGENTS.md` | CLI scripts (to be created with scripts) |
