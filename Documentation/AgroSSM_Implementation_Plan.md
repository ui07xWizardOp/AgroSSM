# AgroSSM: The Complete Implementation Plan

**A Plain-Language, Detailed Guide to Building a Compute-Efficient Agro-Ecosystem AI Model**

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [The Big Idea — In One Sentence](#2-the-big-idea--in-one-sentence)
3. [Why Existing Approaches Fall Short](#3-why-existing-approaches-fall-short)
4. [The AgroSSM Thesis: Structure Over Scale](#4-the-agrossm-thesis-structure-over-scale)
5. [Architecture Overview — The Full Picture](#5-architecture-overview--the-full-picture)
6. [Component-by-Component Deep Dive](#6-component-by-component-deep-dive)
7. [The Training Recipe — How the Model Learns](#7-the-training-recipe--how-the-model-learns)
8. [What We're Predicting — The Five Output Heads](#8-what-were-predicting--the-five-output-heads)
9. [Prerequisites Before Writing a Single Line of Code](#9-prerequisites-before-writing-a-single-line-of-code)
10. [Pilot Scope — What We're Actually Building First](#10-pilot-scope--what-were-actually-building-first)
11. [The Data Pipeline — From Satellites to Tensors](#11-the-data-pipeline--from-satellites-to-tensors)
12. [The Phased Roadmap — 52 Weeks, Sprint by Sprint](#12-the-phased-roadmap--52-weeks-sprint-by-sprint)
13. [Skill Mapping — Who Does What, When](#13-skill-mapping--who-does-what-when)
14. [How We Prove Our Claims — Baselines, Ablations, and Statistics](#14-how-we-prove-our-claims--baselines-ablations-and-statistics)
15. [Risk Register — What Can Go Wrong and How We Handle It](#15-risk-register--what-can-go-wrong-and-how-we-handle-it)
16. [From Pilot to Foundation Model — The Scaling Path](#16-from-pilot-to-foundation-model--the-scaling-path)
17. [Terminology Guardrails — What We Can and Cannot Claim](#17-terminology-guardrails--what-we-can-and-cannot-claim)
18. [Appendix: Decision Log from the Grill-Me Session](#18-appendix-decision-log-from-the-grill-me-session)

---

## 1. What Problem Are We Solving?

Imagine you're a farmer in Iowa growing maize. You need to know:

- **Is my crop stressed right now?** (Crop Stress Detection)
- **When will my crop reach its next growth stage?** (Phenology Forecasting)
- **Will my yield be lower than normal this year?** (Yield-Risk Prediction)

These questions are answered today by individual, disconnected models — one satellite-image model for stress, a weather-based model for phenology, and a statistical model for yield. None of them talk to each other, and none of them understand the *physics* of how water moves through soil, how heat accumulates to drive crop growth, or how a drought that started three months ago is still affecting plants today.

**AgroSSM solves this by building a single AI model that understands all of these processes together**, the way an expert agronomist would. It reads satellite images, weather data, soil maps, and crop calendars simultaneously, and it has a built-in "memory" of how water, heat, and stress accumulate over the growing season — because we bake that knowledge directly into the architecture.

---

## 2. The Big Idea — In One Sentence

> A small, cheap-to-train AI model that understands farming physics from the architecture level can outperform large, expensive "brute-force" models on crop prediction tasks, especially during rare extreme events like droughts.

---

## 3. Why Existing Approaches Fall Short

### 3.1 The "Big Model" Problem

Current state-of-the-art geospatial AI models (called Geospatial Foundation Models, or GeoFMs) like IBM/NASA's **Prithvi** or **DOFA** take a brute-force approach:

1. Collect *millions* of satellite images.
2. Train a *very large* neural network (hundreds of millions of parameters) to reconstruct masked-out patches of those images.
3. Hope that the learned representations are useful for downstream tasks.

**The problem:** These models need **8,000–16,000 A100 GPU-hours** to train. That's roughly $80,000–$160,000 in cloud compute. They also have no understanding of physics — they treat a satellite image of a farm the same way they'd treat a satellite image of an ocean or a city. They don't "know" that rain should increase soil moisture, that heat should accelerate crop growth, or that drought is a *cumulative* process.

### 3.2 The "Separate Everything" Problem

Traditional agricultural monitoring treats each task independently:

| Task | Typical Method | Limitation |
|------|---------------|------------|
| Drought monitoring | SPI/SPEI statistical indices from rainfall data | No spatial resolution; no crop-specific response |
| Crop stress | NDVI anomalies from satellite images | No weather context; can't distinguish drought stress from nutrient stress |
| Phenology tracking | GDD (Growing Degree Days) from temperature | No satellite input; ignores water stress effects on timing |
| Yield prediction | County-level regression on weather + NDVI | Can't handle novel droughts; no physics |

**The problem:** Drought causes stress, stress delays phenology, delayed phenology reduces yield. These are *causally linked* processes, but independent models can't capture those links.

### 3.3 The "False Precision" Problem

Many models naively resample all data to the same 10-meter grid. But weather data (ERA5) comes at 31km resolution. Soil moisture (SMAP) comes at 9km. When you stretch 31km weather data across a 10m grid, every pixel gets the *same* weather value — creating an illusion of precision that doesn't exist.

**AgroSSM avoids this entirely** by keeping each data source at its native resolution and using a smart attention mechanism to let fine-resolution satellite data *query* coarse weather context without pretending the weather varies at 10m.

---

## 4. The AgroSSM Thesis: Structure Over Scale

AgroSSM's central, falsifiable scientific claim is:

> **Process structure can replace scale.** By encoding agronomic knowledge (water balance, heat accumulation, drought memory) directly into the model's architecture, we can achieve comparable or superior performance to large generic models at a *tiny fraction of the compute*.

### 4.1 What "Process Structure" Means in Practice

Instead of building a 134-million-parameter generic transformer and hoping it learns that "rain increases soil moisture," we build a model where:

- **The internal memory state is physically partitioned.** Part of it *must* track soil water, part tracks heat accumulation (AGDD), part tracks crop stress. This isn't a hope — it's a hard architectural constraint.
- **Physics is enforced on the memory, not just the output.** If the water-memory sub-state says soil moisture increased but it didn't rain and there was high evaporation, the model is penalized *during training*. This is novel — prior work only checks physics at the output layer.
- **The model's temporal processing scales linearly (O(n)), not quadratically (O(n²)).** This means longer sequences cost proportionally more, not *exponentially* more.

### 4.2 Why This Is Honest and Defensible

We are **not** claiming to have built a foundation model. We are claiming:

> *"A compute-efficient, process-structured architecture, demonstrated at pilot scale on a single region-crop, with measured throughput and scaling projections toward a full foundation model."*

If a reviewer asks "but can you prove this generalizes globally?", the answer is: "No, but here are measured scaling curves that project the cost. The architectural contribution is the process structure, not the scale."

If our model loses to a brute-force dense model at equal compute, **we report that honestly** — it would mean the thesis is falsified. This is an explicit experiment we run (see Section 14).

---

## 5. Architecture Overview — The Full Picture

Here is the complete data flow, from raw Earth observation inputs to final crop predictions:

```
RAW DATA (Satellite Images, Weather, Soil, Crop Calendars)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  [1] UNIFIED WAVELENGTH-CONDITIONED TOKENIZER       │
│      • One encoder for ALL data types               │
│      • Metadata-driven (wavelength, resolution)     │
│      • Missing-modality handling built in            │
│      Parameters: ~4–8M                              │
└─────────────────────────────────────────────────────┘
     │
     ▼  (produces a sequence of "tokens" — numerical vectors representing each input)
     │
┌─────────────────────────────────────────────────────┐
│  [2] HYBRID TEMPORAL BACKBONE                       │
│      Interleaved blocks that process time:           │
│                                                     │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐           │
│   │ SSM  │→ │ SWA  │→ │ SSM  │→ │ MLA  │→ ...     │
│   └──────┘  └──────┘  └──────┘  └──────┘           │
│                                                     │
│   SSM  = State-Space Model (physical accumulator)   │
│   SWA  = Sliding-Window Attention (local patterns)  │
│   MLA  = Multi-head Latent Attention (coarse data)  │
│      Parameters: ~15–29M                            │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  [3] REGIME-ROUTED MIXTURE-OF-EXPERTS (MoE)         │
│      • 8 tiny specialist sub-networks               │
│      • Only 1 active per token (top-1 routing)      │
│      • Specializes by: irrigated/rainfed,           │
│        vegetative/reproductive, normal/drought      │
│      Total params: ~20–40M, Active: ~3–5M           │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  [4] SHARED AGRO-ECOSYSTEM STATE  z_t               │
│      The SSM hidden state IS the physical memory:   │
│      • h_water  — soil-water balance                │
│      • h_stress — cumulative crop stress            │
│      • h_agdd   — accumulated heat (AGDD)           │
│      • h_free   — unconstrained learned dynamics    │
└─────────────────────────────────────────────────────┘
     │
     ├──────────────────────┬─────────────────────────┐
     ▼                      ▼                         ▼
┌──────────┐  ┌──────────────────┐  ┌───────────────────┐
│ PRIMARY  │  │    AUXILIARY     │  │ PHYSICS-AS-STATE  │
│  HEADS   │  │     HEADS       │  │   REGULARIZER     │
│          │  │                  │  │                   │
│• Stress  │  │ • Drought Traj  │  │ • Water balance   │
│  [4×4]   │  │ • Residual      │  │   on h_water      │
│• Pheno   │  │   Water/Uncert  │  │ • AGDD constraint │
│• Yield   │  │                  │  │   on h_agdd       │
└──────────┘  └──────────────────┘  │ • Stress asymm.  │
                                    │   on h_stress     │
                                    │ • DAG sign prior  │
                                    └───────────────────┘
```

**Total model size:** ~40–80M parameters total, ~25–45M active per token.
**Training cost:** ~30–60 A100-hours (fits in ~300 Google Colab Pro units).

This is an **order of magnitude cheaper** than the v2.3 design (82–134M params, 8,000–16,000 GPU-hours).

---

## 6. Component-by-Component Deep Dive

### 6.1 Component [1]: The Unified Wavelength-Conditioned Tokenizer

#### What It Replaces

The previous design (v2.3) had **four separate encoders**:

| v2.3 Encoder | Input | Params |
|--------------|-------|--------|
| ViT Satellite Encoder | Sentinel-2/Landsat images | 30–50M |
| Temporal Weather Encoder | ERA5 temperature, rain, etc. | 10–15M |
| Soil MLP Encoder | SoilGrids properties | 2–5M |
| Crop Calendar Embedding | Planting dates, crop type | <1M |

Plus a **Scale-Aware Fusion Module** (15–25M params) to combine them.

**Total: ~60–96M parameters just to read and fuse the inputs.**

AgroSSM replaces all of this with **one** tokenizer: **~4–8M parameters**.

#### How It Works (In Plain Language)

Think of it like a universal translator. Instead of having a separate translator for French, German, Spanish, and Mandarin, you have **one translator** that reads a tag on each document ("this is French, formal register, legal domain") and adjusts its translation strategy accordingly.

The tokenizer works the same way. Each input channel — whether it's the Blue band of a satellite image at 10m resolution, or the daily mean temperature from ERA5 at 31km resolution — comes with a **metadata tag**:

```
Channel: "Blue band (490nm)"   →  metadata: {wavelength: 490nm, spatial_res: 10m, temporal_res: 5-day}
Channel: "Mean Temperature"    →  metadata: {variable_id: "T_mean", spatial_res: 31km, temporal_res: daily}
Channel: "Clay percentage"     →  metadata: {variable_id: "clay%", spatial_res: 250m, temporal_res: static}
```

A small **hypernetwork** (a neural network that generates the weights for another neural network) reads this metadata tag and produces the *specific projection weights* for that channel. The result: one shared tokenizer body that can ingest optical satellite bands, weather variables, soil properties, and crop calendar features through a single mechanism.

#### Why This Decision Was Made

1. **Parameter efficiency:** 4–8M instead of 60–96M — a 10× reduction at the input stage alone.
2. **Automatic multi-resolution handling:** The resolution metadata becomes part of the positional encoding, so the model *knows* that a weather token covers a 31km area while a satellite token covers 10m. No forced upsampling.
3. **Extensibility:** Adding a new data source (e.g., SAR radar imagery) requires only defining its metadata tag, not building an entirely new encoder.

#### What Happens When Data Is Missing

Real-world satellite data has gaps — clouds block the view, sensors malfunction, some data products don't exist for older years. The tokenizer handles this with three mechanisms:

1. **Modality-present mask:** A binary flag (1 = present, 0 = missing) attached to each channel.
2. **Learned missing-modality token:** When a channel is absent, a *trainable placeholder vector* is substituted. The model learns what "I don't have this data" means.
3. **Modality dropout during training:** Even when data *is* available, we randomly zero out entire modalities 10–20% of the time. This forces the model to function without any single modality, making it robust to real-world data gaps.

---

### 6.2 Component [2]: The Hybrid Temporal Backbone

This is the architectural heart of AgroSSM. It processes the token sequence over time using three interleaved block types.

#### 6.2.1 SSM Block — The Physical Accumulator (Core Novelty)

**What is a State-Space Model (SSM)?**

Imagine you're tracking the water level in a reservoir. Each day, rain adds water, evaporation removes water, and farmers withdraw water for irrigation. The water level today depends on the water level yesterday plus today's inflows and outflows. This is a *state-space* process — you maintain a running state (water level) and update it incrementally.

An SSM block does exactly this in neural network form. Unlike a standard Transformer (which looks at all time steps simultaneously with O(n²) cost), an SSM processes time steps **sequentially** with O(n) cost, carrying a hidden state forward from one step to the next.

**Why this is perfect for agriculture:**

Drought is not a one-day event. It's a slow *accumulation* — weeks of below-normal rainfall, rising temperatures, declining soil moisture. A drought that started two months ago is still affecting plants today. This accumulation-and-recovery dynamic maps perfectly to an SSM hidden state that carries information forward in time.

**The key novelty — truly partitioned SSM sub-modules:**

Most SSMs treat the hidden state as an opaque blob of numbers. AgroSSM goes further: it uses **4 genuinely independent SSM sub-modules**, each with its own learned recurrence parameters (A, B, C, D, Δ), ensuring each physical quantity is tracked by its own independent dynamics:

| Sub-state | What it tracks | Physical analogy | How it's constrained | Why independent recurrence matters |
|-----------|---------------|------------------|---------------------|------------------------------------|
| `h_water` (via `ssm_water`) | Soil-water memory | The "water level in the reservoir" | Water-balance equation: water in (rain + irrigation + latent inputs) = water out (evaporation + runoff + drainage) + change in storage | Its own A/B/C matrices learn water-specific transition dynamics, uncontaminated by thermal signals |
| `h_stress` (via `ssm_stress`) | Cumulative crop stress | A plant's "damage counter" — stress accumulates faster than it recovers | **Asymmetry prior** (NOT monotonicity): stress increases freely but decreases are penalized 3× more, reflecting that biological damage heals slowly but DOES heal | Its own recurrence learns stress accumulation/recovery rates independently |
| `h_agdd` (via `ssm_agdd`) | Accumulated Growing Degree Days | A "thermal clock" counting how much useful heat the crop has received since planting | AGDD formula: sum of max(0, T_mean - T_base) from planting date; this can only increase; crop stage transitions happen at known AGDD thresholds | Its own recurrence is constrained to strict monotonicity (thermal time never decreases) |
| `h_free` (via `ssm_free`) | Unconstrained learned dynamics | Everything else the model discovers on its own | No constraint — free capacity for the model to learn patterns we haven't thought of | Captures whatever the three constrained sub-states miss |

**Why 4 independent sub-modules instead of slicing a single SSM:**

A single monolithic SSM (like standard Mamba) updates its entire hidden state through shared A, B, C, D matrices. If you simply slice the output and call the slices `h_water` and `h_agdd`, those slices are *entangled* through the shared matrix multiplications — they don't have genuinely independent dynamics. Constraining `h_water` to obey water balance while the recurrence couples it with `h_agdd` creates contradictory gradient signals. AgroSSM avoids this by giving each physical quantity its own independent SSM with its own parameters.

**Why constraining the *state* (not just the output) is novel:**

Previous physics-informed models (PINNs, etc.) apply physics losses at the *output layer* — they check whether the model's *predictions* are physically consistent. AgroSSM goes deeper: it checks whether the model's *internal memory* is physically consistent. This means the temporal dynamics *themselves* are constrained, not just the final answer.

**Analogy:** It's the difference between checking a student's final exam answers (output-only physics) vs. checking the student's *scratch work* to ensure they're using the right method (state-level physics).

#### 6.2.2 SWA Block — Sliding-Window Attention

**What it does:** Captures *local* spatial and short-term temporal patterns in the satellite imagery. A brown patch next to a green patch could be localized crop stress — this requires looking at nearby pixels and recent time steps.

**Why sliding window, not full attention?** Full attention over all tokens at all time steps is expensive (O(n²)). But for *local* patterns, we only need to look at a small neighborhood. Sliding-window attention limits each token to attending to its ~5–7 nearest neighbors in space and time, keeping the cost low.

**Periodic global layers:** Every 4th SWA block uses full attention to capture *seasonal* context — e.g., "is this stress pattern consistent with the overall dry spell across the region?"

#### 6.2.3 MLA Block — Multi-Head Latent Attention for Coarse Context

**The problem it solves:** Satellite imagery is at 10m resolution (many tokens). Weather data is at 31km resolution (few tokens). How do you efficiently let the fine satellite tokens query the coarse weather context?

**The solution:** MLA compresses the coarse Key-Value pairs (weather, soil) into a small **latent bottleneck** before the fine satellite tokens attend to them. Think of it as summarizing a 100-page weather report into a 2-page brief, and then letting each satellite pixel read that brief.

**Why not just full cross-attention?** For our sequence lengths (~30 time steps), full cross-attention would actually be fine computationally. MLA is used here specifically for the *cross-resolution compression* benefit, not for sequence-length scaling.

**What was deliberately NOT used and why:**

| Rejected Technique | Why Rejected |
|---|---|
| DeepSeek-style CSA/HCA | Our sequences are ~30 tokens, not millions. Long-context compression is unnecessary overhead. |
| GQA (Grouped-Query Attention) | An autoregressive decoding optimization. Our model is an encoder, not a text generator. |
| FP4/FP8 quantized experts | Requires NVIDIA Hopper-class GPU kernels not reliably available on Colab. |

#### 6.2.4 How the Blocks Interleave

```
[ SSM ] → [ SWA ] → [ SSM ] → [ MLA ] → [ SSM ] → [ SWA ] → [ SSM ] → [ MLA ] → ...
```

The pattern repeats for 12–16 total blocks. SSM blocks dominate because the accumulator function is the core contribution. SWA and MLA are sprinkled in for local texture and cross-resolution context.

---

### 6.3 Component [3]: Regime-Routed Mixture-of-Experts (MoE)

#### The Insight

Agricultural systems are *regime-switched*. A maize plant in its vegetative stage under normal rainfall behaves completely differently from the same plant in its reproductive stage during a drought. Trying to model both situations with the same parameters is wasteful.

#### How It Works

Instead of one large feed-forward network processing every token, we have **8 small expert networks**. A tiny **router network** looks at each token and decides which expert should process it:

```
Token arrives → Router scores all 8 experts → Top-1 expert is selected → Token processed by that expert only
```

The experts are *not* hard-coded to specific regimes. The router learns soft assignments during training. But we expect interpretable specialization to emerge along these axes:

| Regime Axis | Expected Expert Specialization |
|-------------|-------------------------------|
| Water management | Irrigated fields vs rainfed fields |
| Growth phase | Vegetative stage (growing) vs reproductive stage (fruiting) |
| Climate state | Normal conditions vs drought conditions |

#### Why MoE Instead of a Dense Layer

| Metric | Dense FFN | MoE (8 experts, top-1) |
|--------|-----------|----------------------|
| Total parameters | 20–40M | 20–40M (same) |
| **Active** parameters per token | 20–40M (all) | **3–5M** (1/8th) |
| FLOPs per forward pass | High | Low |
| Paper headline | "20M parameters" | "40M total, 5M active" |

MoE lets us have a *respectable total parameter count* for the paper while keeping *actual compute per token* tiny — exactly the budget lever needed for Colab.

#### Load Balancing

If all tokens get routed to the same expert, the other 7 experts never train and are useless. An auxiliary **load-balance loss** gently penalizes uneven routing, ensuring all experts get trained.

---

### 6.4 Component [4]: The Shared Agro-Ecosystem State (z_t)

This is conceptually the simplest component but perhaps the most important. After the backbone processes a time step, the SSM's hidden state *is* the agro-ecosystem state. It's not an intermediate representation that we throw away — it's the physical quantity we care about.

We decode named quantities from this state:

```python
soil_moisture_estimate = decode_SM(z_t.h_water)     # What's the soil moisture right now?
thermal_time          = decode_AGDD(z_t.h_agdd)      # How much thermal time has accumulated?
stress_level          = decode_stress(z_t.h_stress)   # How stressed is the crop?
```

The output heads then read from this shared state to make their specific predictions.

---

## 7. The Training Recipe — How the Model Learns

Training happens in two stages, following the established practice of *self-supervised pretraining followed by supervised fine-tuning*.

### 7.1 Stage 1: Self-Supervised Pretraining (SSL)

The model first learns general agro-ecosystem representations from *unlabeled* data. Three process-centered pretext tasks are used:

#### SSL-A: Masked Spatiotemporal Reconstruction

**What we do:** Randomly hide 75% of the satellite patches at each time step and 20% of the time steps entirely. Ask the model to reconstruct the hidden parts.

**Why it works:** To reconstruct a missing satellite patch, the model must understand spatial context (nearby fields), temporal continuity (what this field looked like 5 days ago), and cross-modal signals (weather data suggests rain, so NDVI should be rising).

**Analogy:** It's like a jigsaw puzzle where you also remove the box lid — the model must infer what the missing pieces look like from context.

#### SSL-B: Phenology Transition Prediction

**What we do:** Given the current vegetation state, weather history, and crop calendar, predict *when* the next phenological stage transition will happen (e.g., how many days until flowering?).

**Why it works:** This forces the model to learn crop-specific developmental trajectories — that maize transitions from vegetative to reproductive stage after approximately 800 GDD (Growing Degree Days), but this is delayed by water stress.

**AgroSSM-specific mapping:** The prediction is read from the `h_agdd` sub-state, ensuring the thermal accumulator is actually tracking phenological progress.

#### SSL-C: Drought-Stress Trajectory Prediction (Multi-Horizon)

**What we do:** Given recent multi-modal state (weather + satellite + soil moisture), predict vegetation stress anomalies at t+1, t+2, and t+3 (5-day intervals ahead).

**Why it works:** This forces the model to learn drought evolution dynamics — that declining soil moisture today predicts declining NDVI next week, and that the relationship is stronger during reproductive stages.

**AgroSSM-specific mapping:** Predictions are read from `h_water` and `h_stress` sub-states. This is effectively *multi-token prediction reinterpreted for agronomy*.

#### Collapse Monitoring

A known failure mode of self-supervised learning is **representation collapse** — all inputs produce the same output embedding (the model "gives up"). We monitor the covariance matrix of the embeddings and trigger an alert if the effective rank drops (i.e., if the top eigenvalue explains >90% of variance).

### 7.2 Stage 2: Supervised Fine-Tuning

After pretraining, we attach the five output heads and train them on labeled data (crop stress labels, phenology dates, yield statistics) with a combined loss function.

### 7.3 The Total Loss Function

The complete loss during fine-tuning combines everything:

```
L_total = SSL losses (weighted)
        + λ_water(t) × Water-balance state regularizer      ← physics on h_water
        + λ_growth(t) × AGDD growth state regularizer       ← physics on h_agdd
        + λ_stress_asym(t) × Stress asymmetry regularizer        ← asymmetry prior on h_stress (INV-A6)
        + λ_DAG(t) × Temporal-DAG sign consistency prior     ← causal structure
        + λ_balance × MoE load-balance loss                  ← expert utilization
        + λ_calib × Calibration loss                         ← uncertainty quality
```

**Dynamic warm-up:** The physics losses start at zero and ramp up over ~10 epochs. This prevents the physics constraints from destabilizing early training when the model hasn't yet learned basic representations.

---

## 8. What We're Predicting — The Five Output Heads

### 8.1 Primary Supervised Heads (The Paper's Main Results)

| Head | Input (from z_t) | Output | Metric | MVP Target | Stretch Target |
|------|-------------------|--------|--------|------------|----------------|
| **Crop Stress** | h_stress, h_water | Spatial stress probability map [4×4 grid per patch, per time step] + spatial AUROC | AUROC | ≥ 0.90 (in-region), ≥ 0.80 (transfer) | ≥ 0.93, ≥ 0.85 |
| **Phenology Forecast** | h_agdd, z_t | Days until next growth stage transition | MAE (days) | ≤ 5.0 days | ≤ 3.0 days |
| **Yield-Risk** | z_t (full state) | District yield anomaly (% deviation from normal) | R² / RMSE | R² ≥ 0.50, RMSE < 15% | R² ≥ 0.70, RMSE < 10% |

### 8.2 Auxiliary Diagnostic Heads (Supporting Evidence)

| Head | Input | Output | Purpose |
|------|-------|--------|---------|
| **Drought Trajectory** | h_water, h_stress | Short-to-medium term drought progression | Validates that the model's internal drought dynamics are physically plausible |
| **Residual Water / Uncertainty** | h_water, h_free | Latent unobserved water input (I_latent) + predictive uncertainty | Captures unobserved water sources (irrigation, groundwater). **NOT** labeled as "irrigation detection" — it absorbs *all* unexplained water inputs. Also provides calibrated uncertainty estimates (ECE < 0.10). |

---

## 9. Prerequisites Before Writing a Single Line of Code

### 9.1 Software Environment

| Requirement | Specification | Why |
|-------------|---------------|-----|
| Python | 3.10+ | Type hint support, modern async features |
| PyTorch | 2.1+ | Needed for BF16 mixed precision and `torch.compile` |
| PyTorch Lightning | 2.1+ | Handles training boilerplate, multi-GPU, checkpointing |
| Xarray + Rasterio | Latest | Read and align geospatial NetCDF and GeoTIFF data |
| Geopandas + Shapely | Latest | Handle crop-region polygons and spatial operations |
| Weights & Biases | Latest | Experiment tracking, sweep management, artifact logging |
| PyTest | Latest | Unit testing |
| Black + Flake8 + MyPy | Latest | Code formatting, linting, and type checking |

### 9.2 Hardware Requirements

| Resource | Pilot (This Work) | Full Foundation Model (Future) |
|----------|-------------------|-------------------------------|
| GPU | 1× A100-40GB (Colab Pro) | 8× A100-80GB |
| GPU Hours | 30–60 A100-hrs | 8,000–16,000 A100-hrs |
| Storage | ~500 GB – 1 TB | ~2–3 TB |
| RAM | 16–32 GB | 64 GB+ |

### 9.3 Data Access Accounts (Must Be Set Up Before Sprint 1.1)

| Data Source | Where to Register | What You Get |
|-------------|-------------------|--------------|
| Google Earth Engine | [earthengine.google.com](https://earthengine.google.com) | Sentinel-2, Landsat, MODIS, ERA5, CHIRPS, SoilGrids — all from one API |
| ECMWF Climate Data Store | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) | ERA5 / ERA5-Land hourly reanalysis data |
| NASA Earthdata | [earthdata.nasa.gov](https://earthdata.nasa.gov) | SMAP soil moisture, HLS harmonized imagery |
| USDA NASS QuickStats | [quickstats.nass.usda.gov](https://quickstats.nass.usda.gov) | US county-level maize yield statistics |
| USDA CDL | Via GEE or CropScape | US Cropland Data Layer (crop type maps) |

### 9.4 Knowledge Prerequisites

Before implementation, the developer should understand:

1. **Remote sensing basics:** What satellite bands are, what NDVI/EVI vegetation indices measure, how cloud masking works, and what a 5-day composite is.
2. **State-Space Models:** The recurrent formulation `h_t = A·h_{t-1} + B·x_t`, and how Mamba-style selective scan parallelizes this.
3. **Mixture-of-Experts:** How top-k routing works, what load-balance loss is, and why expert collapse is a problem.
4. **Agricultural phenology:** What growth stages maize goes through (emergence → vegetative → silking → grain-fill → maturity), and how GDD drives these transitions.
5. **Water balance equation:** P + I - ET - R - D = ΔSM (precipitation + irrigation - evapotranspiration - runoff - drainage = change in soil moisture).

---

## 10. Pilot Scope — What We're Actually Building First

The full vision includes 4 regions, 2 crops, and 26 years of data. The pilot deliberately restricts scope to prove the architecture works before scaling.

| Parameter | Pilot Value | Why This Choice |
|-----------|-------------|-----------------|
| **Region** | US Corn Belt (Iowa/Illinois) | Cleanest satellite data (few clouds), best yield labels (USDA NASS county-level), well-studied 2012 drought, dense weather station network for validation |
| **Crop** | Maize | Strong phenology signal (clear growth stages visible from space), highly drought-sensitive during silking/grain-fill (July–August), economically critical crop |
| **Temporal era** | Sentinel-2 era only (2017–2023) | Avoids the immense complexity of harmonizing Landsat 5/7/8 imagery with Sentinel-2; provides uniform 10m optical data at 5-day revisit |
| **Patch size** | 64×64 pixels at 10m = ~0.41km × 0.41km | Large enough to capture a few fields; small enough for memory efficiency on Colab |
| **Temporal sequence** | ~24–30 five-day composites per growing season (April–October) | Covers one full growing season; attention over 30 tokens is computationally trivial |
| **Training patches** | 1,000–3,000 representative agricultural patches | Sparse sampling to fit within storage/compute constraints |
| **Extreme holdout** | 2023 drought year | 2012 (iconic drought) is pre-Sentinel-era; 2023 provides an in-era extreme event for testing |

### 10.1 What the Pilot Proves

If AgroSSM works on the US Corn Belt pilot:
- The architectural thesis (process structure > scale) is supported.
- Measured throughput (tokens/sec, hours/epoch) enables credible scaling projections to 4 regions.
- The paper has a defensible contribution even without global coverage.

### 10.2 What the Pilot Does NOT Prove

- Cross-region generalization (that requires the full 4-region matrix — future work).
- Cross-crop transfer (requires training on both wheat and maize — future work).
- Historical drought analysis (the 2012 drought is outside the Sentinel era — documented limitation).

---

## 11. The Data Pipeline — From Satellites to Tensors

### 11.1 The Multi-Resolution Datacube Concept

The datacube is **not** a single dense tensor. It is a multi-resolution structure where each data source lives at its native spatial and temporal resolution:

```
DATACUBE (per patch, per time step)
├── Fine-Resolution Layer (10m, 5-day)
│   └── Sentinel-2 optical bands: Blue, Green, Red, NIR, SWIR1, SWIR2
│       → Shape: [T=30, C=6, H=64, W=64]
│
├── Coarse-Resolution Layer (31km, 5-day)
│   └── ERA5 weather: T_min, T_max, T_mean, Precip, Wind, RH, Solar, ET0
│       → Shape: [T=30, C=8, H=1, W=1]  (one value covers the entire patch)
│
├── Medium-Resolution Layer (250m, static)
│   └── SoilGrids: Clay%, Sand%, OrgC, pH, CEC, BulkDensity, Depth
│       → Shape: [C=7, H=1, W=1]  (static — same for all time steps)
│
├── Crop Context Layer (patch-level, seasonal)
│   └── CDL crop type, planting date, expected harvest date
│       → Shape: [C=3]  (one embedding per patch per season)
│
├── Irrigation Layer (10m or missing, static/annual)
│   └── WorldCereal or GMIA irrigation classification (if available)
│       → Shape: [C=1, H=64, W=64] or modality_present=0
│
└── Metadata
    ├── Spatial coordinates (lat, lon)
    ├── Temporal indices (day-of-year, year)
    ├── Modality-present masks (per channel, per time step)
    └── Cloud/quality flags
```

### 11.2 How Patches Are Selected (Spatiotemporal Sparsity)

We don't download wall-to-wall imagery for all of Iowa and Illinois. Instead:

1. **Obtain CDL crop mask** for each year (2017–2023) to identify maize fields.
2. **Stratified random sampling** of 1,000–3,000 64×64 patches, ensuring:
   - At least 70% of each patch is classified as maize.
   - Representative distribution across counties (so we don't over-sample one area).
   - Mix of irrigated and rainfed patches (if irrigation maps are available).
   - Inclusion of patches near weather stations for validation.
3. **For each patch**, download the full time series of satellite imagery and weather data.
4. **Cloud masking:** For each 5-day composite, compute cloud fraction. If >80% cloudy, mark as missing. If 20–80%, use with a quality weight.

### 11.3 Data Split Design

| Split | Years | Purpose | Notes |
|-------|-------|---------|-------|
| **Training** | 2017–2021 (excluding extreme events) | Learn representations | 5-fold Spatial-Temporal Blocked CV within this pool |
| **Validation** | Cross-fold within 2017–2021 | Hyperparameter tuning | Blocked by county and year to prevent leakage |
| **Test (in-region)** | 2022 | Temporal generalization | Normal year, held out from all training |
| **Test (extreme)** | 2023 | Extreme-event robustness | Drought year, **strictly never seen** during training or validation |

**Zero-leakage protocol:**
- No future satellite observations beyond forecast date in inputs.
- No yield labels from target year during training.
- All normalization parameters computed only from training data.
- Spatial blocks ensure no adjacent patches appear in both train and test.

---

## 12. The Phased Roadmap — 52 Weeks, Sprint by Sprint

### Phase 1: Data Engineering & Pilot Datacube (Weeks 1–14)

> **Goal:** Go from "I have data access accounts" to "I have a clean, validated, split-locked datacube ready for model training."

---

#### Sprint 1.1: Climate & Weather Ingestion (Weeks 1–3)

**What we do:**
- Download ERA5 weather variables (temperature, precipitation, wind, humidity, solar radiation, ET0) for Iowa/Illinois bounding box, 2017–2023, at native ~31km resolution.
- Download CHIRPS precipitation at ~5km resolution for cross-validation against ERA5.
- Aggregate from hourly/daily to 5-day composites aligned with Sentinel-2 revisit.

**Acceptance criteria:**
- Monthly climatology of downloaded data matches published NOAA/CPC values for Iowa.
- No missing 5-day windows in the weather record.

**Verify:** `pytest tests/test_datacube.py -k test_climate_ingestion`

**Compute cost:** CPU-only (Google Earth Engine or CDS API). No GPU hours consumed.

---

#### Sprint 1.2: Satellite & Crop Map Ingestion (Weeks 4–7)

**What we do:**
- Download Sentinel-2 L2A (surface reflectance) imagery for Iowa/Illinois, 2017–2023.
- Compute 5-day composites using best-pixel selection (minimum cloud cover).
- Download USDA CDL crop type maps for each year.
- Download any available irrigation maps (WorldCereal for 2021+; GMIA for static estimates).
- Download SoilGrids data at 250m for the study region.

**Acceptance criteria:**
- Sentinel-2 bands correctly aligned to field boundaries (visual inspection).
- CDL crop mask identifies maize fields with >90% agreement with USDA NASS county statistics.
- No pixel-level spatial misalignment between satellite imagery and crop masks.

**Verify:** Side-by-side visualization of NDVI (from Sentinel-2) overlaid on CDL crop mask.

---

#### Sprint 1.3: Spatiotemporal Sparsity Extraction (Weeks 8–11)

**What we do:**
- Run the stratified random patch selection algorithm (Section 11.2).
- Extract 1,000–3,000 representative 64×64 patches with full time series.
- Apply cloud masking and gap-filling.
- Generate modality-present masks for every channel, every time step, every patch.
- Compute and cache derived vegetation indices (NDVI, EVI, NDWI).

**Acceptance criteria:**
- Total dataset size ≤ 1 TB.
- At least 80% of 5-day composites per patch have <20% cloud contamination.
- Patch distribution covers ≥ 50 distinct Iowa/Illinois counties.

**Verify:** Histogram of patches per county; cloud contamination statistics per patch.

---

#### Sprint 1.4: Dataloader & Split Generator (Weeks 12–14)

**What we do:**
- Implement the multi-resolution PyTorch `Dataset` class that loads a patch and returns the correctly shaped tensors for each resolution tier.
- Generate the 5-fold Spatial-Temporal Blocked cross-validation indices.
- Lock the 2022 test set and 2023 extreme holdout as strictly isolated.
- Run zero-leakage verification: confirm no spatial or temporal overlap between folds.

**Acceptance criteria:**
- Forward pass through a dummy model produces correct tensor shapes.
- No patch ID appears in more than one fold's test set.
- No year appears in both train and test within any fold.

**Verify:** `pytest tests/test_datacube.py -k test_split_leakage`

**Gate G1 (Data Readiness):** All of Phase 1 must pass before we write any model code.

---

### Phase 2: AgroSSM Architecture Implementation (Weeks 15–28)

> **Goal:** Go from "I have data" to "I have a fully connected, forward-pass-verified model architecture."

---

#### Sprint 2.1: Unified Wavelength-Conditioned Tokenizer (Weeks 15–18)

**What we build:**
- The metadata hypernetwork: a small MLP that takes [wavelength/variable_id, spatial_resolution, temporal_resolution] and outputs projection weights.
- The shared tokenizer body: projects each input channel into a common embedding dimension (e.g., 256 or 512).
- Modality-present masking logic and learned missing-modality tokens.

**Acceptance criteria:**
- Tokenizer produces identical output shape regardless of which modalities are present.
- Missing modality tokens are distinct from any real data token (cosine similarity < 0.3).
- Forward pass with all modalities present, with one missing, and with multiple missing — all produce valid (non-NaN, non-infinite) outputs.

**Verify:** `pytest tests/test_encoders.py -k test_unified_tokenizer`

---

#### Sprint 2.2: Hybrid Backbone (SSM + SWA + MLA) (Weeks 19–22)

**What we build:**
- The SSM block with Mamba-style selective scan, with physically partitioned hidden state (h_water, h_stress, h_agdd, h_free).
- The SWA block with configurable window size and periodic global attention.
- The MLA cross-modal block with latent KV compression.
- The interleaving logic: SSM → SWA → SSM → MLA → SSM → SWA → ...

**Acceptance criteria:**
- Full backbone forward pass runs on a single A100-40GB without OOM for batch_size=16.
- SSM hidden state partitions are independently addressable (can extract h_water without affecting h_stress).
- Backbone output shape matches expected dimensions.

**Verify:** `pytest tests/test_encoders.py -k test_hybrid_backbone`

**Key implementation detail:** The SSM block is the hardest part to get right. We will use the `mamba-ssm` library (or a pure-PyTorch re-implementation if licensing is an issue) as the base, and add the physical state partitioning on top.

---

#### Sprint 2.3: Regime-Routed MoE (Weeks 23–25)

**What we build:**
- 8 expert feed-forward networks (small MLPs).
- A router network that scores each token against each expert and selects top-1.
- Load-balance auxiliary loss.
- Expert utilization logging (which expert processes what fraction of tokens).

**Acceptance criteria:**
- All 8 experts receive at least 5% of tokens (no expert collapse).
- Active parameter count per forward pass is ~3–5M (not 20–40M).
- Expert routing distribution shifts between drought and non-drought time steps (interpretability check).

**Verify:** Log expert assignment histograms during a short training run.

---

#### Sprint 2.4: Physics-as-State-Regularizer (Weeks 26–28)

**What we build:**
- Water-balance residual computation on h_water: `residual = decode_SM(h_water_t) - decode_SM(h_water_{t-1}) - (P + I_obs + I_latent - ET - R - D)` where I_obs=0 when irrigation data absent, R via simplified SCS-CN, D via exponential percolation.
- AGDD growth constraint on h_agdd: `loss = ||decode_AGDD(h_agdd_t) - computed_AGDD_t||² + monotonicity_penalty`.
- Stress asymmetry prior on h_stress: `loss = mean(relu(-Δdecode_stress(h_stress)) * asymmetry_factor)` where decreases are penalized 3× more than increases.
- Decode functions (`decode_SM`, `decode_AGDD`, `decode_stress`): `decode_SM` and `decode_AGDD` use `Linear(state_dim, 1) + Softplus` activation (non-negative outputs); `decode_stress` uses `Linear(state_dim, 1)` with NO activation. All are shared across SSM blocks.
- Dynamic lambda scheduling (warm-up from 0 to max over 10 epochs).
- Temporal-DAG sign prior: soft penalty when learned edge signs disagree with agronomic priors.

**Acceptance criteria:**
- Water-balance loss = 0.0 when inputs perfectly satisfy conservation of mass.
- AGDD loss = 0.0 when h_agdd exactly matches hand-computed AGDD from temperature data.
- Monotonicity penalty = 0.0 when AGDD never decreases.
- Dynamic lambda starts at 0 and reaches λ_max after 10 epochs.

**Verify:** `pytest tests/test_physics.py`

**Gate G2/G3:** Model architecture is fully connected and passes all forward-pass tests.

---

### Phase 3: Training & Supervised Heads (Weeks 29–42)

> **Goal:** Go from "I have an architecture" to "I have a trained model that produces useful predictions."

---

#### Sprint 3.1: Self-Supervised Pretraining (Weeks 29–33)

**What we do:**
- Train the full AgroSSM backbone on SSL-A (masked reconstruction), SSL-B (phenology prediction), and SSL-C (drought trajectory prediction) simultaneously.
- Monitor for representation collapse using eigenvalue analysis.
- Checkpoint every epoch (Colab sessions can disconnect at any time).

**Compute budget:** ~15–30 A100-hours. This is the most expensive single step.

**Acceptance criteria:**
- Validation reconstruction loss converges (stops decreasing meaningfully).
- Effective rank of embedding covariance matrix remains > 50% of embedding dimension.
- No NaN/Inf in any gradient or activation.

**Verify:** Weights & Biases training curves.

---

#### Sprint 3.2: Supervised Head Attachment & Fine-Tuning (Weeks 34–38)

**What we do:**
- Attach the 3 Primary heads (Stress, Phenology, Yield-Risk) and 2 Auxiliary heads (Drought Trajectory, Residual Water/Uncertainty).
- Fine-tune the full model end-to-end with the combined loss function (Section 7.3).
- Enable physics-as-state-regularizer with warm-up scheduling.

**Labels used:**
- **Crop Stress:** NDVI anomaly < -1σ or VHI < 35 during the growing season.
- **Phenology:** Transition dates derived from NDVI/EVI time-series inflection points.
- **Yield:** USDA NASS county-level maize yield statistics (converted to anomalies).

**Acceptance criteria:**
- Cross-validation metrics are computed across all 5 folds.
- Stress AUROC ≥ 0.85 on validation (with target ≥ 0.90 on test).
- Phenology MAE ≤ 7 days on validation (with target ≤ 5 on test).

**Verify:** Per-fold metric tables in W&B.

---

#### Sprint 3.3: Hyperparameter Optimization (Weeks 39–42)

**What we do:**
- Sweep learning rates, physics loss weights (λ_water, λ_growth, λ_DAG), modality dropout rate, and MoE routing temperature.
- Use W&B Sweeps with Bayesian optimization.
- Select best configuration based on validation performance averaged across folds.

**Compute budget:** ~5–10 A100-hours for a small sweep (10–20 runs).

**Gate G4/G5:** SSL and physics contributions are measured and validated.

---

### Phase 4: Rigorous Validation & Claims Audit (Weeks 43–52)

> **Goal:** Go from "I have a trained model" to "I have a defensible paper with honest, statistically rigorous results."

---

#### Sprint 4.1: Statistical Bootstrapping (Weeks 43–46)

**What we do:**
- Compute all metrics on the 2022 test set and 2023 extreme holdout.
- Run **Spatial Block Bootstrap** (50km × 50km blocks) with 1,000 resamples to compute 95% confidence intervals for all metrics.
- Run **Year-Level Leave-One-Out** analysis within the training pool to assess temporal stability.
- Apply Bonferroni correction for multiple comparisons across metrics.

**Why this matters:** Standard random-sample confidence intervals are *inflated* for geospatial data because nearby pixels are correlated. Block bootstrap accounts for this spatial autocorrelation. Without it, a reviewer would (rightly) reject the paper.

---

#### Sprint 4.2: Extreme-Event & Baseline Evaluation (Weeks 47–49)

**What we do:**
- Evaluate on the 2023 extreme drought holdout. Compare to all baselines.
- Run all baseline models (Climatology, Persistence, GDD-phenology, SPI/SPEI/VHI, XGBoost, LSTM/TCN) on the same data with the same splits.
- Run claim-critical ablations (Section 14).

---

#### Sprint 4.3: Physical & Causal Consistency Audits (Weeks 50–51)

**What we do:**
- Verify that water-balance RMSE is ≥15% lower with physics-as-state-regularizer vs without.
- Verify that downstream metrics degrade by ≤3% when physics is enabled.
- Check temporal-DAG sign consistency against agronomic priors (target ≥90%).
- Check I_latent correlation with irrigation maps where available (p < 0.05).

---

#### Sprint 4.4: Manuscript & Reproducibility Package (Week 52)

**What we do:**
- Produce all paper tables and figures.
- Measure and report throughput (tokens/sec, hours/epoch) for scaling projections.
- Freeze code, configs, splits, and checkpoints for reproducibility.
- Write the manuscript.

**Gate G7 (Research Readiness):** At least one claim is statistically significant. All claims are reported (including negative results).

---

## 13. Skill Mapping — Who Does What, When

These are the AI agent skills (specialist subagent personas) that will be invoked at each phase:

| Skill | What It Does | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-------|-------------|---------|---------|---------|---------|
| **`data-quality-auditor`** | Profiles datasets, validates masks, checks for anomalies, audits split integrity | Sprint 1.1 (climate QA), 1.2 (satellite QA), 1.3 (cloud masking), 1.4 (split leakage) | — | Sprint 3.1 (modality dropout verification) | — |
| **`senior-ml-engineer`** | Implements PyTorch modules, dataloaders, training loops, mixed precision, checkpointing | Sprint 1.4 (dataloader) | Sprint 2.1 (tokenizer), 2.2 (backbone), 2.3 (MoE) | Sprint 3.1–3.3 (training pipeline) | — |
| **`senior-computer-vision`** | Designs ViT-style patch processing, spatial attention, masked reconstruction | — | Sprint 2.2 (SWA block, spatial tokenization) | Sprint 3.1 (SSL-A masked reconstruction) | — |
| **`senior-data-scientist`** | Implements physics losses, statistical tests, causal diagnostics, bootstrap methods | — | Sprint 2.4 (physics regularizer) | — | Sprint 4.1 (block bootstrap), 4.2 (baselines), 4.3 (physics/causal audit) |
| **`autoresearch-agent`** | Runs hyperparameter sweeps, manages W&B experiments, computes scaling projections | — | — | Sprint 3.3 (HP sweeps) | Sprint 4.2 (baseline comparison), 4.4 (scaling curves) |

---

## 14. How We Prove Our Claims — Baselines, Ablations, and Statistics

### 14.1 Required Baselines (Compute-Aware)

| # | Baseline | Type | Compute |
|---|----------|------|---------|
| 1 | Historical Mean (Climatology) | Minimum skill | ~Free |
| 2 | Persistence (predict last observed value) | Temporal | ~Free |
| 3 | GDD Phenology Model | Simple crop-physics | ~Free |
| 4 | SPI/SPEI/VHI Drought Indices | Statistical drought monitoring | ~Free |
| 5 | XGBoost on hand-crafted features | Classical ML | ~1–2 CPU-hours |
| 6 | LSTM / TCN | Deep temporal | ~3–6 A100-hours |
| 7 | Prithvi-EO-2.0 / DOFA | GeoFM (if feasible) | Report "not evaluated" if budget exceeded |

### 14.2 Claim-Critical Architecture Ablations

These are the experiments that directly test the AgroSSM thesis:

| Ablation | What We Swap | What It Tests | Claim |
|----------|-------------|---------------|-------|
| **SSM → Attention** | Replace SSM blocks with standard transformer blocks | Is the linear-time state-space core actually better than quadratic attention? | Efficiency thesis |
| **State physics → Output physics** | Move water-balance/AGDD losses from h_water/h_agdd to output layer only | Is constraining the *internal memory* better than constraining only the *predictions*? | **Core novelty** |
| **MoE → Dense FFN** | Replace 8 experts with one dense layer of equal active params | Does regime specialization help, or is it just the parameter count? | Efficiency thesis |
| **Unified tokenizer → Separate encoders** | Replace the single tokenizer with v2.3's four separate encoders | Is unification actually better, or just cheaper? | Efficiency thesis |
| **AgroSSM vs Small Dense Quadratic** | Train a small dense transformer with the same data and compute budget | **THE DECISIVE EXPERIMENT:** Can a generic model match AgroSSM at equal resources? | **Central falsifiable claim** |

The last row is the one that matters most. If the dense model wins, the thesis is falsified, and we report that honestly.

### 14.3 Statistical Protocol

- **Confidence intervals:** 95% CI from Spatial Block Bootstrap (50km blocks, 1,000 resamples).
- **Significance testing:** Paired bootstrap test; reported as p-values with Bonferroni correction.
- **What is prohibited:** Naive pixel-level random-sample t-tests (these inflate significance on correlated geospatial data).

---

## 15. Risk Register — What Can Go Wrong and How We Handle It

| Risk | Severity | What Happens | Mitigation | Fallback |
|------|----------|-------------|------------|----------|
| **SSL collapse** | High | All embeddings converge to the same vector; model learns nothing useful | Monitor eigenvalue/effective-rank of embedding covariance; temperature scheduling | Revert to generic MAE (He et al. 2022) |
| **Physics destabilizes training** | Medium | Water-balance or AGDD loss causes NaN gradients or training divergence | Dynamic warm-up (start λ=0, ramp over 10 epochs); gradient clipping | Remove physics losses entirely; report model without physics as baseline |
| **MoE expert collapse** | Medium | All tokens get routed to 1–2 experts; other experts are wasted | Load-balance auxiliary loss; monitor expert utilization | Replace MoE with dense FFN (ablation becomes the main model) |
| **Colab session disconnects** | High | GPU session terminates mid-training; hours of compute wasted | Aggressive checkpointing (every epoch); resume-from-checkpoint logic | Pre-purchase reserved GPU time on Lambda Labs or similar |
| **Insufficient labeled data** | Medium | Too few crop-stress or phenology labels to train heads effectively | Use pseudo-labels from NDVI anomalies; semi-supervised training | Focus on SSL pretraining quality; report downstream heads as exploratory |
| **2023 is not actually extreme enough** | Low | The 2023 drought may not be severe enough to differentiate models | Document severity metrics; compare stress distributions 2023 vs normal years | Report as limitation; note that 2012 (the truly extreme year) is outside Sentinel era |
| **Compute budget exceeded** | Medium | Sweeps or ablations take more hours than available | Prioritize claims C1–C3 (core novelty); reduce patch count to 1,000 | Report subset of ablations; mark others as "future work" |

---

## 16. From Pilot to Foundation Model — The Scaling Path

The pilot is not the end goal. It's a proof of concept. Here's how the architecture scales:

| Dimension | Pilot (This Work) | Full Foundation Model (Future) |
|-----------|-------------------|-------------------------------|
| Regions | 1 (US Corn Belt) | 4+ (US, India, Spain, Italy) |
| Crops | 1 (Maize) | 2+ (Maize, Wheat) |
| Temporal era | 2017–2023 (Sentinel only) | 2000–2026 (requires EO harmonization) |
| Active parameters | 25–45M | Scale by adding experts and depth |
| Compute | 30–60 A100-hrs | Projected linearly from measured pilot throughput |

**The scaling contribution:** We report measured throughput (tokens/sec, hours/epoch) at pilot scale. A reviewer can then independently verify whether our scaling projection to the full FM is realistic. This turns a compute limitation into a *credible scaling-law contribution* — which is a publishable result in itself.

---

## 17. Terminology Guardrails — What We Can and Cannot Claim

| ✅ Allowed | ❌ Prohibited | Why |
|-----------|-------------|-----|
| Crop-climate risk | Food security risk | We don't model socioeconomic factors |
| Yield anomaly / yield shortfall | Food shortage / famine prediction | Far beyond our scope |
| I_latent (latent residual water input) | Irrigation detection | I_latent captures ALL unexplained water, not just irrigation |
| Process-centered SSL | Self-supervised learning (generic) | Must specify what kind of SSL |
| Weak physics regularization | Physics-informed model | "Weak" acknowledges our physics is approximate |
| Temporal DAG constraint (structural prior) | Causal discovery | We encode known relationships, not discover new ones |
| High directional consistency (≥90%) | 100% sign consistency | Biology is non-monotonic |
| Demonstrated at pilot scale | Foundation model | We haven't trained at foundation scale |

---

## 18. Appendix: Decision Log from the Grill-Me Session

These are the resolved decisions from our interactive design review:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | 3 or 5 output heads? | 3 Primary + 2 Auxiliary | Harmonizes all documents; drought trajectory and residual water are diagnostic, not primary supervised tasks |
| Q2 | Irrigation as required input or optional? | Required where available (v2.3 refined to "preferred covariate") | AgroSSM v3.0 pilot uses irrigation when available; modality dropout ensures robustness when absent |
| Q3 | Temporal scope 2017–2023 or 2000–2026? | v3.0 pilot: 2017–2023 (Sentinel only); full FM: 2000–2026 | Pilot avoids EO harmonization complexity; scaling projections cover the full scope |
| Q4 | Maize in one region or two? | Two (US Corn Belt + Po Valley, Italy) in full FM; one (US Corn Belt) in pilot | De-confounds crop-region effects; pilot proves concept on one |
| Q5 | "All Resolved" wording? | "Resolved in design; pending empirical validation" | Honest about what's been done (design) vs what hasn't (experiments) |
| Q6 | Temporal splits? | Train 2017–2021, Test 2022, Extreme Holdout 2023 | Clean chronological split with blocked cross-validation within training pool |
| Q7 | Statistical protocol? | Spatial Block Bootstrap (50km) + Year-Level LOO | Standard for geospatial data; avoids inflated significance |
| Q8–12 | Bibliography, TOC, licensing, compute fallback, architecture diagrams | All approved as recommended | Documentation hygiene items |
