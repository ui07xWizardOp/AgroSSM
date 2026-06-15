# AgroEarthFM Architecture v3.0 — "AgroSSM"

## A Compute-Efficient, Process-Structured Architecture for Agro-Ecosystem Representation Learning

**Document Version:** 3.0 (Proposed — Efficiency Breakthrough)
**Date:** 2026-06-11
**Status:** Architecture Proposal — For Review
**Supersedes (architecture only):** v2.3 dense multi-encoder design
**Classification:** Research Prototype Architecture Specification

---

## 0. Reading Guide and Relationship to Prior Specs

This document **replaces only the architecture and compute strategy** of the frozen v2.3 MVRC spec. It does **not** change the scientific framing, the claim register (C1–C6), the data licensing, the no-leakage protocol, the evaluation metrics, or the ethical/terminology constraints. Those remain governed by `AgroEarthFM_MVP_Spec_v2.3.md`.

What changes here:

1. The four separate modality encoders + heavy fusion block are replaced by a **single unified wavelength-conditioned tokenizer**.
2. The two attention-based temporal-memory transformers are replaced by a **hybrid linear-time state-space backbone** (SSM + sliding-window attention + latent cross-modal attention).
3. Dense feed-forward layers are replaced by a **tiny regime-routed Mixture-of-Experts (MoE)**.
4. Weak physics is reformulated as a **state-regularizer on the SSM hidden state** rather than a separate output-space loss only.
5. The compute claim changes from "train a foundation model (8k–16k A100-hours)" to **"demonstrate a compute-efficient process-structured architecture at pilot scale (~30–60 A100-hours) with explicit scaling projections."**

---

## 1. The Central Thesis: Process Structure Replaces Scale

The dominant paradigm in geospatial foundation models is **scale**: large dense transformers pretrained on enormous unlabeled Earth-observation corpora. This is effective but compute-prohibitive (the v2.3 design alone projected 8,000–16,000 A100-hours).

AgroSSM proposes the opposite thesis:

> **A small, process-structured, linear-time architecture that bakes agronomic process priors directly into its inductive biases can match or exceed large quadratic geospatial models on transfer and extreme-event agro-ecosystem tasks, at a tiny fraction of the compute.**

The architectural contribution and the efficiency are the *same idea*. The model is cheap **because** it encodes process structure (drought accumulation, phenological progression, water balance) into the architecture itself, rather than forcing a generic large model to rediscover that structure from data.

### 1.1 Why This Is a Legitimate From-Scratch Contribution

- It is trained **from scratch** (no frozen backbone, no prior-work weights).
- The novelty is in the **module design and their composition**, not in scale.
- Every efficiency technique is **scientifically motivated by the agronomy**, not copied from LLMs by analogy.
- The claim is falsifiable: if a large dense GeoFM beats AgroSSM at equal data, the thesis fails (this is an explicit ablation).

### 1.2 Honest Claim Boundary (Non-Negotiable)

This work **does not** claim to have trained a foundation model. The honest, defensible framing is:

> *"A compute-efficient, process-structured architecture, demonstrated at pilot scale on a single region-crop, with parameter, active-FLOPs, and scaling projections toward a full foundation model."*

Prohibited claims: "trained foundation model", "global generalization", "state-of-the-art GeoFM" (unless directly measured against one at equal data). These would be desk-reject risks given pilot-scale compute.

---

## 2. Pilot Scope (Trainable on 300 Colab Pro Units)

Confirmed scope floor for the trainable demonstration:

| Parameter | Pilot Value | Rationale |
|---|---|---|
| Region | **US Corn Belt (Iowa/Illinois)** | Cleanest clouds, best labels (USDA NASS county-level), well-documented 2012 drought, dense weather network |
| Crop | **Maize** | Strong phenology signal; drought-sensitive at silking/grain-fill |
| Era | **Sentinel era only (2017–2023)** | Avoids the EO-harmonization burden of the pre-2015 Landsat era; uniform 10–20m optical |
| Patch size | **64×64 at 10m** | ~0.41 km × 0.41 km; primary config |
| Sequence length | **~24–30 five-day composites** | One growing season; full attention over this length is effectively free |
| Extreme holdout | **2012 is pre-era; use 2023 drought as extreme holdout** | 2012 excluded by Sentinel-era scope; document this limitation |

**Compute ceiling:** Google Colab Pro, ~300 compute units ≈ **30–60 A100-40GB-hours** (or a few hundred T4/L4-hours). The entire architecture below is sized to fit this.

---

## 3. Architecture Overview

```text
Multi-Resolution Inputs (optical, weather, soil, crop-calendar, irrigation*)
        |
[1] Unified Wavelength-Conditioned Tokenizer  (ONE encoder for ALL modalities)
        |   -> modality-present masks + learned missing-modality tokens
        |
[2] Hybrid Temporal Backbone (interleaved blocks):
        |     - SSM block (Mamba-style)  : linear-time physical accumulator (CORE)
        |     - SWA block (sliding window): local spatio-temporal optical structure
        |     - MLA cross-modal block    : latent-compressed coarse context (weather/soil)
        |
[3] Regime-Routed MoE feed-forward (top-1, ~8 experts) : capacity at low active compute
        |
[4] Shared Agro-Ecosystem State  z_t  (SSM hidden state = physical memory)
        |
[5] Output Heads: Spatial Crop Stress [4x4] | Phenology | Yield-Risk | Drought Traj | Residual-Water/Uncertainty
        |
[6] Physics-as-State-Regularizer (water balance + AGDD on z_t) + Temporal-DAG sign prior
```

---

## 4. Component [1]: Unified Wavelength-Conditioned Tokenizer

Replaces the four separate encoders (satellite ViT + weather transformer + soil MLP + calendar embedding) **and** the scale-aware fusion block with **one** shared tokenizer.

### 4.1 Design

Following the unified-encoder trend (Gemma-style unified encoding; DOFA's wavelength-conditioned dynamic weights), each input channel is described by **metadata** rather than hard-coded to an encoder:

| Metadata field | Example values |
|---|---|
| Central wavelength / variable id | 490nm (Blue), 842nm (NIR), "precip", "T_mean", "clay%", "irrigation" |
| Native spatial resolution | 10m, 31km, 250m |
| Native temporal resolution | 5-day, daily, static |
| Modality-present flag | 1 / 0 |

A small **hypernetwork** maps each channel's metadata to that channel's projection weights, so a single tokenizer body ingests optical bands, weather variables, soil properties, and calendar features through one shared mechanism.

### 4.2 Multi-Resolution Handling (Preserved from v2.x)

- **No forced 10m upsampling.** Coarse variables produce few tokens with a resolution tag in their positional encoding.
- Coarse context enters the backbone through the **MLA cross-modal block** (Section 5.3), not by spatial replication.

### 4.3 Missing-Modality Protocol (Preserved)

- Per-channel modality-present mask.
- Learned missing-modality token.
- Modality dropout p=0.1–0.2 during training.

### 4.4 Parameter Budget

| Item | Params |
|---|---|
| Wavelength/metadata hypernetwork | ~1–3M |
| Shared tokenizer body | ~3–5M |
| **Subtotal** | **~4–8M** |

*Savings vs v2.3: removes ~60–90M (four encoders + fusion block collapse into ~4–8M).*

---

## 5. Component [2]: Hybrid Temporal Backbone

The backbone interleaves three block types. This is the architectural heart of AgroSSM.

### 5.1 SSM Block (Mamba-style) — The Physical Accumulator [CORE NOVELTY]

State-space models scale **O(n)** in sequence length and carry an explicit recurrent hidden state. The key contribution:

> **The SSM hidden state is structured to represent physical accumulators**: soil-water memory, cumulative stress, and accumulated growing degree days (AGDD).

This aligns the architecture with the agronomy: drought is a long-memory accumulation-and-recovery process, and an SSM hidden state is a natural accumulator. We use **4 truly independent SSM sub-modules**, each with its own A, B, C, D, Δ parameters:

| Hidden sub-state | Physical meaning | Regularized by |
|---|---|---|
| `h_water` (via `ssm_water`) | Soil-water memory / balance | Water-balance residual (Section 7.1) |
| `h_stress` (via `ssm_stress`) | Cumulative crop stress | Stress-asymmetry prior (INV-A6) |
| `h_agdd` (via `ssm_agdd`) | Accumulated thermal time | AGDD growth constraint (Section 7.2) |
| `h_free` (via `ssm_free`) | Unconstrained learned dynamics | (none — free capacity) |

**Why this is novel:** prior work applies physics losses only at the *output*. AgroSSM applies them to the *recurrent state*, so the temporal dynamics themselves are physically constrained. This is the central, falsifiable architectural claim.

### 5.2 SWA Block (Sliding-Window Attention) — Local Optical Structure

Fine optical spatio-temporal structure is local. A sliding window over fine tokens (with periodic global layers for seasonal context) captures local stress patches and texture cheaply.

| Parameter | Value |
|---|---|
| Window size | ~5–7 time steps / local spatial neighborhood |
| Global layers | 1 in every 4 blocks |

### 5.3 MLA Cross-Modal Block — Latent-Compressed Coarse Context

Multi-Head Latent Attention compresses coarse weather/soil Key-Value tensors into a **low-dimensional latent KV**. Fine optical tokens (queries) attend to this compact latent context. This is the right efficiency tool for *this* problem (multi-resolution context), as opposed to million-token machinery.

> **Design note:** We deliberately do **not** adopt DeepSeek-style CSA/HCA million-token compression or GQA decode optimizations. Sequences here are ~30 tokens; quadratic attention over 30 tokens is negligible. Importing long-context machinery would be unjustified complexity. MLA is adopted only for cross-resolution KV compression, where it is genuinely motivated.

### 5.4 Block Interleaving Pattern

```text
[ SSM ] -> [ SWA ] -> [ SSM ] -> [ MLA ] -> [ SSM ] -> [ SWA ] -> ... (xN)
```

SSM blocks dominate (the accumulator core); SWA and MLA are interleaved for local structure and cross-modal context.

### 5.5 Parameter Budget

| Item | Params (depth ~12–16) |
|---|---|
| SSM blocks | ~8–15M |
| SWA blocks | ~4–8M |
| MLA cross-modal blocks | ~3–6M |
| **Subtotal** | **~15–29M** |

---

## 6. Component [3]: Regime-Routed MoE

Feed-forward layers are replaced by a **fine-grained, top-1 Mixture-of-Experts** where experts specialize by **agro-climatic regime**.

### 6.1 Regime Routing (Scientifically Motivated)

Agro-ecosystems are regime-switched. Candidate routing regimes (the router learns soft assignments; these are interpretability anchors, not hard rules):

| Axis | Regimes |
|---|---|
| Water management | irrigated / rainfed |
| Phenophase | vegetative / reproductive |
| Climate state | normal / drought |

### 6.2 Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Num experts | 8 | Small; covers regime combinations |
| Routing | top-1 | Minimizes active compute |
| Active experts/token | 1 | ~1/8 of expert params active |
| Load balancing | auxiliary load-balance loss | Prevent expert collapse |

### 6.3 Capacity vs Active Compute

| Quantity | Value |
|---|---|
| Total expert params | ~20–40M |
| **Active** expert params/token | ~3–5M |

*MoE gives a respectable total parameter count for the paper while keeping active compute tiny — exactly the budget lever needed for Colab.*

---

## 7. Component [6]: Physics-as-State-Regularizer

Reformulates weak physics from output-only losses to **hidden-state constraints** (plus the existing output checks).

### 7.1 Water Balance on `h_water`

```text
residual_t = decode_SM(h_water_t) - decode_SM(h_water_{t-1})
             - (P_t + I_obs_t + I_latent_t - ET_t - R_t - D_t)
L_water = w_data * || residual_t ||^2
```

The SSM water sub-state must evolve consistently with the water-balance equation. `I_obs_t` is 0.0 when data is missing. R and D are explicitly computed (SCS-CN for runoff, exponential decay for drainage). `decode_SM` uses Softplus for non-negativity and is shared across all SSM blocks. `I_latent` is inferred (latent residual water input; NOT "irrigation detection", per v2.3 policy).

### 7.2 AGDD Growth on `h_agdd`

```text
GDD_t = max(0, T_mean_t - T_base)        # T_base: maize = 10C
AGDD_t = AGDD_{t-1} + GDD_t
L_growth = || decode_AGDD(h_agdd_t) - AGDD_t ||^2
         + monotonicity_penalty(h_agdd)   # AGDD cannot decrease
```

`decode_AGDD` uses Softplus for non-negativity and is shared across all SSM blocks.

### 7.3 Stress Asymmetry on `h_stress`

Unlike thermal time (which is strictly non-decreasing), crop stress can recover. However, recovery is biologically slower than stress onset (damage takes time to heal). This is modeled via a stress asymmetry prior:

```text
delta_h_stress_t = decode_stress(h_stress_t) - decode_stress(h_stress_{t-1})
L_stress_asym = mean(relu(-delta_h_stress_t) * asymmetry_factor)
```

Where:
- `asymmetry_factor` = 3.0 (default; stress decreases are penalized 3x more than increases).
- `relu(-delta)` selects only decreasing steps (recovery).
- `decode_stress` uses a `Linear(state_dim, 1)` layer with NO activation (stress is a latent quantity that can be negative during recovery phases) and standard Xavier uniform initialization. It is shared across all SSM blocks.

### 7.4 Temporal-DAG Sign Prior (Preserved, Soft)

Unchanged from v2.3: temporal DAG is a structural prior / inductive bias, **not** causal discovery. Sign-consistency target: **>=90%** consistent with agronomic priors (not 100%; biological systems are non-monotonic).

### 7.5 Total Loss

```text
L_total = Sum_i alpha_i * L_SSL_i              # SSL-A/B/C (from v2.3)
        + lambda_water(t) * L_water           # state-regularizer (dynamic warmup)
        + lambda_growth(t) * L_growth         # state-regularizer (dynamic warmup)
        + lambda_stress_asym(t) * L_stress_asym    # stress asymmetry prior (INV-A6)
        + lambda_DAG(t) * L_temporal_DAG      # soft sign prior
        + lambda_balance * L_moe_loadbalance  # MoE load balancing
        + lambda_calib * L_calibration
```

Dynamic lambda scheduling and warm-up are preserved from v2.3.

---

## 8. SSL Objectives (Preserved from v2.3, Mapped to AgroSSM)

| ID | Objective | AgroSSM mapping |
|---|---|---|
| SSL-A | Masked spatiotemporal reconstruction | Tokenizer + backbone reconstruct masked optical tokens |
| SSL-B | Phenology transition prediction | Read from `h_agdd` + state → transition timing |
| SSL-C | Drought-stress trajectory (multi-horizon) | Read from `h_water`/`h_stress` → t+1,t+2,t+3 (this *is* multi-token prediction, agro-reinterpreted) |

Collapse monitoring (eigenvalue / effective-rank) preserved.

---

## 9. Total Parameter and Active-Compute Budget

| Component | Total Params | Active Params/token |
|---|---|---|
| Unified tokenizer | 4–8M | 4–8M |
| Hybrid backbone (SSM/SWA/MLA) | 15–29M | 15–29M |
| Regime MoE | 20–40M | 3–5M (top-1) |
| Output heads | 2–5M | 2–5M |
| **TOTAL** | **~40–80M total** | **~25–45M active** |

Headline number for the paper: ~40–80M total parameters, **~25–45M active** — an order of magnitude below the v2.3 dense 82–134M design, with linear-time temporal scaling.

---

## 10. Compute Feasibility Projection (Mapped to 300 Colab Units)

Assumptions: single A100-40GB session; BF16/FP16 mixed precision (FP4/FP8 **not** assumed — not reliably available on Colab); 64×64 patches; ~30 time steps; 1,000–3,000 representative patches; one region-crop.

| Stage | Est. A100-hours | Notes |
|---|---|---|
| Datacube preprocessing | CPU-bound (Colab CPU/Earth Engine) | Not GPU-billed |
| Baselines (RF/XGBoost on CPU; LSTM/TCN small) | ~3–6 | Climatology/persistence are ~free |
| SSL pretrain (SSL-A/B/C, small model, sparse patches) | ~15–30 | Linear-time backbone keeps this low |
| Supervised head training + joint loss | ~5–10 | |
| Physics + ablations (reduced set) | ~5–10 | |
| Evaluation / bootstrap / OOD | ~2–5 | |
| **Total** | **~30–60 A100-hours** | **Fits ~300 Colab units, but tight** |

### 10.1 Discipline Required (This Budget Is Not Forgiving)

- Train on **sparse representative patches**, never dense full scenes.
- Checkpoint aggressively; Colab sessions disconnect.
- Run baselines and preprocessing on **CPU** to conserve GPU units.
- Reduce the ablation matrix to the **claim-critical** ablations only (see 11.2).
- Use gradient accumulation for effective batch size on 40GB.

### 10.2 Mixed Precision Honesty

Claim **BF16/FP16 mixed precision**. Mention FP8/FP4 (DeepSeek-style) and the Muon optimizer as **future work**, not as implemented. Use **AdamW**.

---

## 11. Baselines and Ablations to Prove the Efficiency Thesis

### 11.1 Required Baselines (Subset of v2.3, Compute-Aware)

Climatology, Persistence, GDD-phenology, SPI/SPEI/VHI, XGBoost, LSTM/TCN. (GeoFM baselines Prithvi/DOFA: report "not evaluated due to compute constraints" if they exceed budget — honest and acceptable.)

### 11.2 Claim-Critical Architecture Ablations (The Paper's Core Evidence)

| Ablation | Tests | Claim |
|---|---|---|
| SSM core → replaced by attention | Value of linear-time state core | Efficiency thesis |
| Physics-as-state-regularizer → output-only physics | Value of state-level constraints | Core novelty |
| Regime MoE → dense FFN | Capacity-vs-active-compute tradeoff | Efficiency thesis |
| Unified tokenizer → separate encoders (v2.3 style) | Value of unification | Efficiency thesis |
| Full model vs **small dense quadratic** at equal data/compute | Direct test of "structure replaces scale" | **Central falsifiable claim** |

The last row is the decisive experiment: if a small dense quadratic model matches AgroSSM at equal data and compute, the thesis is falsified — and you report that honestly.

---

## 12. What Was Deliberately NOT Adopted (and Why)

Reviewers respect knowing what *not* to use. Explicitly rejected:

| Technique | Reason rejected for this problem |
|---|---|
| DeepSeek CSA/HCA million-token compression | Sequences are ~30 tokens; long-context machinery is unjustified |
| GQA | An autoregressive-decode optimization; irrelevant to this encoder workload |
| FP4 expert precision | Needs Hopper-class kernels not reliable on Colab |
| Muon optimizer | Unproven in this domain; AdamW is the safe choice (Muon = future work) |
| mHC (manifold residual connections) | Matters at hundreds of layers; this model is ~12–16 deep |
| Configurable reasoning / think-tags | An LLM-agentic feature; meaningless for a geospatial regression/representation model |
| Frozen pretrained GeoFM backbone | User requires a from-scratch architectural contribution |

---

## 13. Scaling Projection (Pilot → Foundation Model)

To preserve the foundation-model vision honestly, document the scaling path **without claiming it was executed**:

| Axis | Pilot (this work) | Projected full FM |
|---|---|---|
| Regions | 1 (US Corn Belt) | 4+ (per v2.3 matrix) |
| Crops | 1 (maize) | 2+ (maize, wheat) |
| Era | Sentinel (2017–2023) | 2000–2026 (+ EO harmonization) |
| Active params | 25–45M | scale experts/depth |
| Compute | 30–60 A100-hrs | project linearly from measured pilot throughput |

Report **measured throughput (tokens/sec, hours/epoch)** at pilot scale and extrapolate. This turns a compute limitation into a credible scaling-law contribution.

---

## 14. One-Line Architecture Definition

> AgroSSM is a from-scratch, compute-efficient, process-structured agro-ecosystem model that fuses all modalities through a single wavelength-conditioned tokenizer, processes them with a linear-time hybrid state-space backbone whose hidden state encodes physical water/stress/AGDD accumulators, and routes capacity through a tiny regime-specialized Mixture-of-Experts — demonstrating, at pilot scale, that process structure can replace scale for crop-stress, phenology, and yield-risk prediction.

---

## 15. Open Decisions for Reviewer (You)

1. Confirm depth (~12 vs ~16 blocks) and SSM:SWA:MLA ratio once first throughput measurements exist.
2. Confirm number of MoE experts (8 proposed) after a load-balance smoke test.
3. Confirm whether to attempt one GeoFM baseline (Prithvi/DOFA) if budget permits, or declare out-of-scope up front.

---

*End of AgroEarthFM Architecture v3.0 (AgroSSM) Proposal.*
