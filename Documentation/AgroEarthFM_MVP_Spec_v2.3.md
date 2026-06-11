# AgroEarthFM: A Physics-Regularized Multimodal Foundation Model for Climate-Resilient Agro-Ecosystem Modeling

## Research Prototype Specification v2.3

**FROZEN — Publication-Grade Year-1 MVRC Specification**

**Document Version:** 2.3 (Frozen)
**Date:** 2026-05-31
**Status:** Pre-Implementation Specification — Validated & Frozen
**Classification:** Research Prototype Specification
**Supersedes:** v2.2 (2026-05-29)

---

## Revision Log

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | 2026-05-22 | Complete Finalized Specification from Scope-Lock session |
| v2.1 | 2026-05-22 | Draft incorporating grill-me session changes |
| v2.2 | 2026-05-29 | Comprehensive internal review: fixed irrigation policy, data splits, data availability by era, missing success criteria, Sprint 4.4, Po Valley verification, project structure, I_latent clarification, storage/compute updates, baselines, dynamic scheduling, label noise, AGDD formulas, gates, risk register |
| v2.3 | 2026-05-31 | **External review freeze revision**: (1) Reframed paper as climate-resilience foundation model; (2) Irrigation reworded as "preferred covariate where available" with explicit coverage reporting; (3) Added EO Harmonization Layer (HLS preferred) for cross-sensor stationarity; (4) Replaced "100% DAG sign consistency" with "high directional consistency with agronomic priors"; (5) Split all success criteria into MVP targets and stretch goals; (6) Added explicit 4-dimensional OOD definitions (geography, year/extreme, sensor mix, crop/region transfer); (7) Formalized 3 locked SSL pretext tasks (SSL-A/B/C); (8) Clarified water-balance residual formulation; (9) Lowered yield MVP target to R-squared >= 0.50; (10) Expanded baseline suite with Prithvi-EO-2.0, DOFA, U-Net, Temporal ViT; (11) Added uncertainty quantification metrics (ECE, CRPS); (12) Added 128x128 patch-size ablation; (13) Added key citations; (14) Clarified causal DAG as structural prior, not full causal discovery |

---

## 1. Objective

Build a multimodal, spatiotemporally sparse, physics-regularized, and causally informed machine-learning prototype for crop stress detection, crop phenology forecasting, and agricultural yield-risk/anomaly prediction.

The central contribution of AgroEarthFM is not merely agricultural prediction, but **learning climate-resilience representations through process-centered self-supervision**. Drought dynamics, vegetation stress, crop phenology, and yield anomalies serve as downstream evaluations of this core representation. This framing elevates the work from a domain-specific prediction model to a **foundation model for agro-ecosystem resilience**.

The goal is to deliver a Year-1 Minimum Viable Research Contribution (MVRC) that demonstrates:

1. **C1 (Joint Modeling):** Joint drought + phenology representation improves performance over single-task baselines.
2. **C2 (Multimodal Fusion & OOD Robustness):** Multi-source input improves robustness under explicit out-of-distribution (OOD) conditions — defined across four dimensions: (a) geographic transfer, (b) year and extreme-weather holdouts, (c) sensor-mix variation, and (d) crop/region cross-transfer.
3. **C3 (Process SSL):** Process-centered Self-Supervised Learning (three locked pretext tasks) outperforms general masked autoencoders on transfer and extreme-event metrics.
4. **C4 (Physics Regularization):** Weak water-balance constraints reduce physical inconsistencies (residual water-budget RMSE) without degrading downstream metric performance.
5. **C5 (Causal Diagnostics):** Structured temporal DAG constraints, used as an inductive bias (not full causal discovery), yield biologically plausible and temporally consistent representations.
6. **C6 (Tail Events):** Robustness under extreme weather events, sensor-mix variation, and spatial-temporal block holdouts, including cross-region and cross-crop transfer.

### 1.1 MVRC Deliverable Definition

The Year-1 MVRC consists of three deliverables:

| Deliverable | Description | Completion Criteria |
|-------------|-------------|---------------------|
| **Research Paper** | Manuscript-ready paper with ablation tables, error analysis, and validated claims C1–C6 | At least 1 claim with statistically significant positive result; all claims reported (positive or negative) |
| **Code Repository** | Fully reproducible codebase with frozen dependencies, documented configs, and fixed train/val/test splits | pytest passes; any experiment in the paper reproducible from config + splits |
| **Model Checkpoint** | Final trained model checkpoint with evaluation scripts | Checkpoint loads and produces reported metrics within 1% tolerance |

### 1.2 Claim Validation Priority

| Priority | Claims | Justification |
|----------|--------|---------------|
| Must-validate | C1, C2, C3 | Core novelty of the paper |
| Strongly-validate | C4, C6 | Key differentiators |
| Validate-if-possible | C5 | High-risk, high-reward; acceptable as interpretability contribution |

### 1.3 Explicit Out-of-Distribution (OOD) Definitions

Claims C2 and C6 require precise OOD definitions. The following four OOD dimensions are evaluated:

| OOD Dimension | Definition | Test Protocol | Example |
|---------------|-----------|---------------|---------|
| **Geographic OOD** | Model evaluated on a region entirely held out from training | Train on 3 regions; test on the 4th | Train: IGP + Spain + US; Test: Po Valley |
| **Temporal / Extreme-Weather OOD** | Model evaluated on years or extreme events held out from training | Test on extreme drought holdouts (2005 Spain, 2012 US, 2022 IGP) | 2012 US Corn Belt drought (~30% yield loss) |
| **Sensor-Mix OOD** | Model evaluated with different sensor configurations than training | Train on Sentinel-2 era (2017+); test on Landsat-only era (pre-2015) | Landsat 5/8 inputs only, no Sentinel-2 |
| **Crop/Region Transfer OOD** | Model evaluated on a crop-region pair that combines unseen crop in unseen region | Cross-crop cross-region transfer | Train wheat in IGP+Spain, maize in US+Italy; test wheat-in-US or maize-in-IGP |

---

## 2. Revised Core Hypothesis

> A multimodal temporal representation trained on joint drought and crop-phenology process objectives, and regularized by weak water-balance constraints, will improve crop-stress detection, phenology forecasting, and yield-risk prediction under out-of-region, out-of-year, and extreme-event validation compared with single-domain, non-physics, and conventional machine-learning baselines.

### 2.1 Hypothesis Scope Boundary

The hypothesis does NOT claim:
- Full causal discovery from observational data alone
- Food-security risk prediction (requires socioeconomic data not included)
- Global generalization across all climate zones
- Decision-grade counterfactual estimates
- Field-level yield accuracy from administrative-level yield labels

---

## 3. Tech Stack & Dependencies

- **Programming Language:** Python 3.10+
- **Core ML Framework:** PyTorch 2.1+ / PyTorch Lightning 2.1+
- **Data Processing:** Xarray, Rasterio, Pandas, NumPy, Scipy, TorchGeo
- **Spatial Analysis:** Geopandas, Shapely
- **Logging & Visualization:** Weights & Biases, Matplotlib, Seaborn
- **Testing & Quality:** PyTest, Flake8, Black, MyPy

---

## 4. Commands

### 4.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Operational Commands

- **Run Data Preprocessing & Sparsity Extraction:**
  ```bash
  python scripts/preprocess_data.py --config config/data_config.yaml
  ```

- **Run Unit Tests (Strict Local Spec):**
  ```bash
  pytest tests/ --cov=src/ --cov-report=term-missing
  ```

- **Run Training (Multi-GPU/Single-GPU):**
  ```bash
  python scripts/train_model.py --config config/model_config.yaml --fold 0
  ```

- **Run Diagnostic Validation & Bootstrap Audits:**
  ```bash
  python scripts/evaluate_model.py --checkpoint paths/checkpoint.ckpt
  ```

- **Code Formatting & Linting:**
  ```bash
  black src/ tests/ scripts/
  flake8 src/ tests/ scripts/
  mypy src/
  ```

---

## 5. Project Structure

```text
agro_fm_mvp/
├── config/
│   ├── data_config.yaml            # Data paths, crop regions, variables
│   ├── model_config.yaml           # Hyperparameters, head configs, losses
│   └── ssl_config.yaml             # SSL objective configs, masking ratios, schedules
├── Documentation/
│   └── AgroEarthFM_MVP_Spec_v2.3.md  # This specification document
├── scripts/
│   ├── preprocess_data.py          # Sparsity extraction, EO harmonization, datacube builder
│   ├── train_model.py              # Main training script (PyTorch Lightning)
│   └── evaluate_model.py           # Evaluation, bootstrapping, and causal audits
├── src/
│   ├── datacube/
│   │   ├── __init__.py
│   │   ├── datacube_loader.py      # Spatial-temporal patch dataset loading
│   │   ├── patch_extractor.py      # 64x64 / 128x128 patch extractor with ablation support
│   │   ├── eo_harmonization.py     # Cross-sensor spectral harmonization (HLS alignment)
│   │   └── modality_masks.py       # Missing-modality token and modality dropout logic
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── satellite_encoder.py    # ViT-based harmonized optical encoder (Landsat + Sentinel-2)
│   │   ├── weather_encoder.py      # Temporal weather transformer encoder
│   │   ├── soil_encoder.py         # Static MLP encoder for soil/topography
│   │   └── calendar_encoder.py     # Crop stage prior embedding
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── cross_attention.py      # Scale-aware cross-attention layer
│   ├── ssl/
│   │   ├── __init__.py
│   │   ├── ssl_a_reconstruction.py # SSL-A: Masked spatiotemporal reconstruction
│   │   ├── ssl_b_phenology.py      # SSL-B: Phenology transition prediction
│   │   ├── ssl_c_stress_trajectory.py # SSL-C: Drought-stress trajectory prediction
│   │   └── collapse_monitor.py     # Eigenvalue-based representation collapse detection
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── primary_heads.py        # Crop Stress, Phenology, Yield Risk heads
│   │   └── auxiliary_heads.py      # Drought Trajectory, Residual Water/Uncertainty heads
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── constraints.py          # Weak water-balance and AGDD growth losses
│   │   └── dynamic_scheduler.py    # Dynamic lambda scheduling for physics loss weights
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── causal_dag.py           # Temporal DAG constraint and sign auditing
│   │   ├── bootstrap.py            # Spatial block bootstrap & Year LOO bootstrap
│   │   └── uncertainty.py          # ECE, CRPS, prediction interval coverage
│   └── baselines/
│       ├── __init__.py
│       ├── climatology.py          # Historical mean baseline
│       ├── persistence.py          # Temporal persistence baseline
│       ├── gdd_phenology.py        # GDD-based phenology model
│       ├── drought_indices.py      # SPI/SPEI/VHI drought index models
│       ├── tree_models.py          # Random Forest / XGBoost baselines
│       ├── lstm_tcn.py             # LSTM / TCN temporal deep learning baselines
│       ├── unet_baseline.py        # U-Net spatial segmentation baseline
│       ├── temporal_vit.py         # Temporal ViT baseline (non-process SSL)
│       └── geofm_baselines.py      # Prithvi-EO-2.0, DOFA (if feasible)
└── tests/
    ├── __init__.py
    ├── test_datacube.py            # Patch loading, missing modalities, zero-leak, harmonization
    ├── test_encoders.py            # Dimension and tensor flow tests
    ├── test_physics.py             # Water-balance constraints, AGDD formulas
    ├── test_diagnostics.py         # Autocyclicity, gradient leak, uncertainty metrics
    └── test_ssl.py                 # Collapse monitor, masking correctness, objective convergence
```

---

## 6. Architectural Specifications

### 6.1 Output Heads

**Primary Supervised Downstream Heads:**

| Head | Output | Primary Metric | MVP Target | Stretch Target |
|------|--------|---------------|------------|----------------|
| **Crop Stress Head** | Probability of crop stress (water/heat) and spatial IoU | AUROC | >= 0.90 in-region; >= 0.80 transfer | >= 0.93 in-region; >= 0.85 transfer |
| **Phenology Forecast Head** | Expected timing of next phenological transition | MAE in days | <= 5.0 days | <= 3.0 days |
| **Yield-Risk Head** | District/regional yield anomaly percentage | R-squared / RMSE | R-squared >= 0.50; RMSE < 15% | R-squared >= 0.70; RMSE < 10% |

**Auxiliary Diagnostic Heads:**

| Head | Output | Primary Metric | MVP Target | Stretch Target |
|------|--------|---------------|------------|----------------|
| **Drought Trajectory Head** | Short-to-medium term drought progression anomalies | RMSE (SPI/SPEI anomaly) | >= 10% relative RMSE reduction vs persistence | >= 20% relative RMSE reduction |
| **Residual Water/Uncertainty Head** | Predictive uncertainty and I_latent inference | Correlation with irrigation maps + calibration (ECE, CRPS) | I_latent correlates with irrigated areas (p < 0.05); ECE < 0.10 | ECE < 0.05; CRPS competitive with ensemble baselines |

### 6.2 Irrigation Input Policy (Frozen from v2.2, Refined in v2.3)

> **Irrigation is a preferred covariate where available; when absent, the model uses a dedicated missing-modality token and reports the coverage limitation explicitly.**

Specifically:

1. Where irrigation maps exist (ESA WorldCereal or equivalent regional sources, when available), they enter the model as an **input modality** with a modality-present binary mask and learned missing-modality placeholder token.
2. Where irrigation maps are unavailable, the modality-present mask is set to 0 and the learned missing-modality token is used.
3. The model is NOT restricted to regions with irrigation map availability; it can operate on any region in the Crop-Region Matrix.
4. Coverage statistics (what fraction of patches/eras have irrigation data) must be reported in the Data Readiness Report.
5. **I_latent** (Latent Residual Water-Input Inference) remains the key mechanism for capturing unobserved water inputs:

   > I_latent: Residual water input NOT captured by available irrigation maps — includes unreported irrigation, groundwater contribution, capillary rise, and measurement error in known irrigation estimates.

6. During training, **modality dropout** (probability p=0.1-0.2) randomly zeros out entire modalities (including irrigation maps when available) to ensure the model can function without any single modality.

**Rationale:** Irrigation mapping is still region-specific and data availability is uneven across the study period (2000-2026). ESA WorldCereal provides 10m irrigation maps only from 2021 onward; coarser products (GMIA, IWMI) provide static estimates. Making irrigation a required input would severely constrain model applicability and is not defensible for the full temporal scope.

### 6.3 EO Harmonization Layer (NEW in v2.3)

Because the study spans 2000-2026, optical inputs come from multiple sensors with different spectral characteristics: Landsat 5 TM (7 bands), Landsat 7 ETM+ (8 bands), Landsat 8/9 OLI (11 bands), and Sentinel-2 MSI (13 bands). Without harmonization, the satellite encoder receives inconsistent spectral distributions over time, violating stationarity assumptions.

**Harmonization Strategy:**

1. **Preferred approach:** Use NASA Harmonized Landsat-Sentinel (HLS) products (HLSL30/HLSS30) which provide radiometrically and geometrically harmonized surface reflectance at 30m from 2015 onward (Claverie et al. 2018).
2. **Pre-HLS era (2000-2014):** Apply a linear spectral transformation to map Landsat 5/7 bands into the HLS spectral space, following established cross-sensor calibration (Chastain et al. 2019; Flood 2017). This transformation is fit on overlap periods and validated on held-out years.
3. **All optical inputs are transformed into a harmonized spectral representation prior to encoder ingestion.** The encoder receives a consistent band set regardless of source sensor.
4. The harmonization layer is implemented in `src/datacube/eo_harmonization.py` and tested in `tests/test_datacube.py`.

| Parameter | Specification |
|-----------|---------------|
| Output spectral space | HLS-compatible: Blue, Green, Red, NIR, SWIR1, SWIR2 (6 bands minimum) |
| Harmonization method | HLS native (2015+); linear cross-sensor calibration (2000-2014) |
| Calibration fit period | 2015-2016 overlap (Landsat 8 + Sentinel-2) |
| Validation | Band-wise RMSE < 0.02 reflectance units on held-out overlap data |

### 6.4 Causal Chain and Temporal DAG (Frozen — with clarified framing)

The temporal DAG is a **structural prior and inductive bias**, NOT a full causal discovery method. It does not claim to identify causal effects from observational data; it encodes known agronomic relationships as soft constraints.

The revised causal structure is time-indexed, eliminating same-time-step cycles:

```
Weather(t) -> ET(t)
Rainfall(t), ET(t), Soil(t), I_latent(t) -> Soil_Moisture(t)
Soil_Moisture(t), Temperature(t), Crop_Stage(t) -> Vegetation_Health(t)
Vegetation_Health(t), AGDD(t), Crop_Calendar -> Crop_Stage(t+1)
Vegetation_Health(t), Crop_Stage(t), Stress_Accumulation(t) -> Yield_Risk(t+1)
```

**DAG Structural Properties:**

| Property | Specification |
|----------|---------------|
| Acyclicity | No same-time directed cycles; temporal indexing ensures DAG validity |
| Edge directionality | All edges are temporally forward or same-time with clear causal precedence |
| Vegetation Health -> Crop Stage | Forward in time: VH(t) -> CS(t+1) |
| Crop Stage -> Vegetation Health | Same time step: CS(t) -> VH(t) (crop stage determines stress sensitivity) |
| I_latent role | Latent residual water input (unobserved irrigation + groundwater + capillary rise + measurement error) enters soil moisture equation at time t |
| Stress accumulation | Cumulative term that accumulates across time steps |
| Sign consistency | **High directional consistency with agronomic priors where applicable** — NOT 100% absolute consistency (see Section 16 for criteria) |

**Why not 100% sign consistency:** Real agro-ecosystems exhibit non-monotonic responses. For example, temperature increase may accelerate growth in early season but cause heat stress during anthesis. A rigid 100% sign-consistency requirement is biologically unrealistic. Instead, we report the fraction of sign-consistent edges and investigate violations as model anomalies.

### 6.5 Scope, Spatiotemporal Sparsity, & Regional Matrix

- **Temporal Coverage:** 2000-2026 (see Section 8 for Data Availability by Era).
- **Crop-Region Matrix (Balanced & De-confounded):**

  | Crop | Region 1 | Region 2 |
  |------|----------|----------|
  | Wheat | Indo-Gangetic Plain (India) | Iberian Peninsula (Spain) |
  | Maize | US Corn Belt (Iowa/Illinois) | Po Valley (Italy) |

- **Sparsity Optimization:** Training utilizes **Spatiotemporal Sparsity**. The system extracts and trains only on 1,000-5,000 representative 64x64 agricultural patches across each region.

- **Patch-Size Ablation (NEW in v2.3):** Drought context may require larger spatial context than a single 64x64 field patch. The spec mandates:

  | Configuration | Patch Size | Context Area | Status |
  |---------------|-----------|--------------|--------|
  | Baseline | 64x64 at 10m | ~0.41 km x 0.41 km | Primary training |
  | Ablation | 128x128 at 10m | ~1.28 km x 1.28 km | Must be evaluated for sensitivity |

  If the 128x128 ablation shows significant improvement (>5% relative on primary metrics), it becomes the recommended configuration for the paper.

### 6.6 Data Splits & Temporal Blocked K-Fold

- **Chronological Split:**
  - **Train/Validation Pool:** 2000-2021
  - **In-Region Test set:** 2023-2025 (excluding extreme drought years for the relevant region)
  - **Extreme Drought Years (Held out entirely for extreme-event evaluation):** 2005 (Spain), 2012 (US Corn Belt), 2022 (IGP)

| Region-Crop | In-Region Test Years | Extreme Drought Holdout |
|-------------|---------------------|------------------------|
| IGP Wheat | 2023-2025 | 2022 |
| US Corn Belt Maize | 2023-2025 | 2012 |
| Iberian Peninsula Wheat | 2023-2025 | 2005 |
| Po Valley Maize | 2023-2025 | (none specifically held out) |

- **Validation Protocol:** Within the 2000-2021 pool, hyperparameters are tuned using **5-Fold Spatial-Temporal Blocked Cross-Validation** to eliminate spatial and temporal leakage (Brenning 2012; Karasiak et al. 2021).
- **Split locking rule:** All splits defined BEFORE model selection. Fixed and committed to version control.

### 6.7 Autocorrelation & Statistical Protocol

- Naive pixel-level random splits and t-tests are prohibited.
- Performance statistics (confidence intervals, p-values) are calculated using a **Spatial Block Bootstrap (50km x 50km blocks)** and a **Year-Level Leave-One-Out (LOO) Bootstrap** (Brenning 2012).
- If multiple metrics are reported, correct for multiple comparisons (e.g., Bonferroni or Benjamini-Hochberg).

### 6.8 No-Leakage Protocol for Forecasting Tasks

1. Future satellite observations beyond forecast date are PROHIBITED in inputs.
2. Full-season composites in early-season predictions are PROHIBITED.
3. Yield labels or statistics from the target prediction period are PROHIBITED.
4. Gap-filled products that use future observations are PROHIBITED unless explicitly documented.
5. Region-level normalization using held-out region statistics is PROHIBITED.
6. All preprocessing (normalization parameters, cloud thresholds) is fit only on training data.

### 6.9 Scale Compliance Rule

Predictions are validated ONLY at the spatial scale of the ground truth labels. If labels are district-level, predictions are aggregated to district level before computing yield metrics.

---

## 7. Complete Model Architecture

### 7.1 MVP Architecture (Frozen)

```
EO Harmonization Layer (Landsat/Sentinel-2 -> unified spectral space)
        |
Fine Satellite Encoder (ViT-based, harmonized optical input)
        |
Weather / Climate Temporal Encoder (Temporal Transformer)
        |
Soil / Static Context Encoder (MLP)
        |
Crop Calendar / Crop Type Embedding (Learned Embedding)
        |
Irrigation Map Encoder (preferred covariate; modality-present mask + missing-modality token)
        |
Scale-Aware Fusion Module (Cross-Attention with Resolution Awareness)
        |
Hierarchical Temporal Memory Module (5-day + Monthly)
        |
Shared Agro-Ecosystem State Representation
        |
 +----------------+----------------+----------------+
 | Crop Stress    | Phenology       | Yield Risk /   |
 | Head           | Forecast Head   | Yield Anomaly  |
 +----------------+----------------+----------------+
        |
 +----------------+----------------+
 | Drought        | Residual Water/ |
 | Trajectory     | Uncertainty     |
 | Head           | Head (I_latent) |
 +----------------+----------------+
        |
Weak Physics Regularization (Water Balance + AGDD Growth) during training
Temporal DAG Constraints (structural prior, inductive bias — NOT full causal discovery)
```

### 7.2 Component Specifications

#### 7.2.1 EO Harmonization Layer

| Parameter | Specification |
|-----------|---------------|
| Input | Landsat 5/7/8/9 reflectance (2000-2014) or Sentinel-2 reflectance (2015+) or HLS (2015+) |
| Output | Harmonized 6-band surface reflectance in HLS spectral space |
| Method | HLS native (2015+); linear cross-sensor calibration (2000-2014) |
| Calibration fit | 2015-2016 Landsat-8/Sentinel-2 overlap |
| Validation | Band-wise RMSE < 0.02 reflectance units |

#### 7.2.2 Fine Satellite Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | Vision Transformer (ViT) |
| Input | Harmonized optical reflectance (6-band HLS-compatible) |
| Patch size | 16x16 pixels |
| Temporal input | Sequence of 5-day composites |
| Positional encoding | 2D spatial + temporal encoding + sensor-era indicator |
| Output | Token sequence per time step |
| Pretraining | SSL-A: Masked spatiotemporal reconstruction |
| Approximate parameter count | 30-50M parameters |

#### 7.2.3 Weather / Climate Temporal Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | Temporal Transformer |
| Input | ERA5 / CHIRPS climate variables at native resolution (~31km / ~5km) |
| Variables | Temperature (min/max/mean), precipitation, wind speed, solar radiation, humidity, ET0 |
| Temporal resolution | Daily aggregated to 5-day windows |
| Positional encoding | Temporal encoding with day-of-year and season indicators |
| Output | Climate state vector per time step |
| Approximate parameter count | 10-15M parameters |

#### 7.2.4 Soil / Static Context Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | MLP |
| Input | SoilGrids variables at ~250m resolution |
| Variables | Clay content, sand content, organic carbon, pH, CEC, bulk density, soil depth |
| Output | Static soil-context vector |
| Approximate parameter count | 2-5M parameters |

#### 7.2.5 Crop Calendar / Crop Type Embedding

| Parameter | Specification |
|-----------|---------------|
| Architecture | Learned embedding layer |
| Input | Crop type (one-hot or embedding index), planting date, expected harvest date |
| Source | GEOGLAM crop calendars, crop masks |
| Output | Crop-context vector per pixel/patch |
| Approximate parameter count | <1M parameters |

#### 7.2.6 Irrigation Map Encoder (Preferred Covariate)

| Parameter | Specification |
|-----------|---------------|
| Architecture | MLP with modality-present mask |
| Input | Irrigation layers from WorldCereal or equivalent regional sources, when available |
| Missing modality handling | Binary modality-present mask; learned missing-modality placeholder token |
| Modality dropout | p=0.1-0.2 during training (randomly zero out entire irrigation modality) |
| Output | Irrigation-context vector per pixel/patch |
| Approximate parameter count | 1-2M parameters |

#### 7.2.7 Scale-Aware Fusion Module

| Parameter | Specification |
|-----------|---------------|
| Architecture | Cross-attention with resolution-aware positional encoding |
| Input | Token sequences from all encoders |
| Mechanism | Fine-resolution tokens attend to coarse-resolution context |
| Resolution handling | Multi-resolution tokens preserved; no forced upsampling to 10m |
| Missing modality | Modality-present mask + learned missing-modality token |
| Output | Fused multimodal representation per spatial-temporal position |
| Approximate parameter count | 15-25M parameters |

#### 7.2.8 Hierarchical Temporal Memory Module

| Parameter | Specification |
|-----------|---------------|
| Architecture | Two-level temporal transformer |
| Level 1 | Short-term memory: 5-day composite sequence (captures within-season dynamics) |
| Level 2 | Long-term memory: Monthly aggregated sequence (captures seasonal patterns) |
| Input | Fused multimodal token sequence |
| Output | Temporal state representation at each time step |
| Approximate parameter count | 20-30M parameters |

### 7.3 Model Size and Token Budget

| Component | Estimated Parameters | Token Budget |
|-----------|---------------------|--------------|
| EO Harmonization Layer | <1M | — (preprocessing) |
| Satellite Encoder (ViT) | 30-50M | ~256 tokens per time step |
| Weather Encoder | 10-15M | ~1-4 tokens per time step |
| Soil Encoder | 2-5M | ~1-4 tokens per time step |
| Crop Calendar Embedding | <1M | ~1 token per time step |
| Irrigation Map Encoder | 1-2M | ~1-4 tokens per time step |
| Fusion Module | 15-25M | Cross-attention queries: fine tokens; keys/values: all tokens |
| Temporal Memory | 20-30M | 5-day: ~24 tokens; Monthly: ~6 tokens |
| Output Heads | 2-5M | Per-head MLP |
| **Total MVP** | **~82-134M** | **~285-295 tokens per time step** |

---

## 8. Data Strategy

### 8.1 Data Availability by Era

| Era | Years | Fine Optical | Coarse Optical | SAR | Weather | Soil Moisture | Irrigation Maps | Yield Labels |
|-----|-------|-------------|----------------|-----|---------|---------------|-----------------|--------------|
| **Pre-Sentinel** | 2000-2012 | Landsat 5 TM (30m, 16-day); Landsat 7 ETM+ (SLC-off from May 2003, ~22% data loss) | MODIS (250m-1km, daily) | Not available | ERA5 (31km) | Not available (pre-SMAP) | Not available | Available |
| **Landsat Transition** | 2013-2016 | Landsat 8 OLI (30m, 16-day); Landsat 7 ETM+ (SLC-off) | MODIS | Sentinel-1 (from 2014, limited) | ERA5 | SMAP (from 2015, limited) | Not available | Available |
| **Sentinel Era** | 2017-2020 | Sentinel-2 (10m, 5-day) + Landsat 8; HLS (from 2015) | MODIS | Sentinel-1 (full) | ERA5 + ERA5-Land | SMAP + ESA CCI | Partial (GMIA, IWMI) | Available |
| **Modern Era** | 2021-2026 | Sentinel-2 + Landsat 8/9 + HLS | MODIS | Sentinel-1 | ERA5 + ERA5-Land | SMAP + ESA CCI | ESA WorldCereal (from 2021) | Available |

### 8.2 Multi-Resolution Datacube Design

**Core Principle:** The model does NOT force all data to a common 10m grid. Each variable is represented at its appropriate spatial and temporal scale.

**Resolution Tiers:**

| Tier | Spatial Scale | Temporal Scale | Data Types | Token Treatment |
|------|---------------|----------------|------------|-----------------|
| Fine | 10-30m | 5-day composites | Harmonized optical reflectance, vegetation indices | Fine-resolution visual/vegetation tokens |
| Intermediate | 100-500m | 5-day to monthly | Aggregated vegetation dynamics, SAR-derived moisture proxies | Aggregated context tokens |
| Coarse | 1-10km | Daily to monthly | ERA5/CHIRPS weather, SMAP/ESA CCI soil moisture, drought indices | Coarse temporal weather/moisture tokens |
| Regional | Admin/field level | Seasonal/annual | Yield statistics, management records, crop maps | Regional context tokens |

### 8.3 Complete Data Source Catalog

#### Satellite Imagery

| Source | Product | Resolution | Access | License |
|--------|---------|------------|--------|---------|
| HLS (preferred) | HLSL30/HLSS30 | 30m | LP DAAC / GEE | Public domain |
| Sentinel-2 | L2A reflectance | 10-20m | Copernicus / GEE | CC-BY-SA 4.0 |
| Landsat 5/7/8/9 | L2 reflectance | 30m | USGS / GEE | Public domain |

#### Climate Data

| Source | Variables | Spatial Resolution | Temporal Resolution | Access | License |
|--------|-----------|-------------------|---------------------|--------|---------|
| ERA5 (ECMWF) | T_min, T_max, T_mean, P, wind, RH, solar radiation, ET0 | ~31km | Hourly -> daily -> 5-day | CDS API / GEE | CC-BY-4.0 |
| ERA5-Land | Enhanced land variables | ~9km | Hourly | CDS API | CC-BY-4.0 |
| CHIRPS | Precipitation (1981-present) | ~5km | Daily | CHIRPS API / GEE | Public domain (Funk et al. 2015) |

#### Soil Data

| Source | Variables | Resolution | Access |
|--------|-----------|------------|--------|
| SoilGrids 250m | Clay %, Sand %, Organic C, pH, CEC, Bulk density, Depth | ~250m | ISRIC / GEE (CC-BY 4.0) |

#### Soil Moisture

| Source | Product | Resolution | Depth | Access |
|--------|---------|------------|-------|--------|
| SMAP | L3 Enhanced | ~9km (enhanced ~3km) | 0-5cm | NASA DAAC / GEE |
| ESA CCI Soil Moisture | Active + Passive merged | ~25km | 0-5cm | ESA CCI / GEE |

#### Crop Calendar and Crop Mask

| Source | Variables | Resolution | Access |
|--------|-----------|------------|--------|
| GEOGLAM Crop Calendar | Planting/harvest dates | Regional/country | GEOGLAM / USDA |
| ESA WorldCereal | Crop type + irrigation classification | 10m | ESA (open-source; Van Tricht et al. 2023) |
| CDL (US) | Crop type | 30m | USDA NASS |
| ISRO Crop Maps (India) | Crop type, kharif/rabi | State/district | Government of India / ISRO |

#### Yield Labels

| Source | Scale | Access | Quality |
|--------|-------|--------|---------|
| USDA NASS (US) | County-level | Public | High quality |
| ICRISAT (India) | District-level | Public | Good quality |
| Eurostat/CAP (EU) | NUTS2/Province | Public | Good quality |
| ISTAT (Italy) | Province/Region | Public (with registration) | Good quality |

#### Irrigation and Management Proxies

| Source | Data Type | Resolution | Access | Era Availability |
|--------|-----------|------------|--------|-----------------|
| GMIA (FAO) | Irrigated area fraction | ~10km | FAO | All eras (static) |
| IWMI Global Irrigation Map | Irrigated/rainfed classification | ~500m | IWMI | All eras (static) |
| ESA WorldCereal | Irrigation classification | 10m | ESA | Modern era only (2021+) |
| USDA FRIS (US) | Irrigation practices | Admin | USDA | Most eras |
| Regional Registries (Po Valley) | Irrigation district boundaries | Admin | Regione Lombardia/Emilia-Romagna | Most eras |

### 8.4 Study Region Data Availability Verification

(As verified in v2.2 — all 4 regions confirmed with explicit data-availability tables. No changes needed.)

### 8.5 Phenology Pseudo-label Quality & Noise Estimation

Phenology transitions are derived from NDVI/EVI time-series derivatives. These are noisy pseudo-labels, not ground truth.

**Required noise estimation:**
1. Compare phenology transition dates from different indices (NDVI vs EVI vs NDWI); report standard deviation.
2. Where available, compare with ground observations (Pan European Phenology Network for Spain/Italy; USDA for US).
3. Report label noise statistics in the Data Readiness Report.
4. Use soft labels with associated uncertainty during training.

### 8.6 Crop Stress Labels

Crop stress labels are defined as vegetation health anomalies below a threshold:
- NDVI anomaly < -1 standard deviation, OR
- VHI < 35, OR
- Documented drought event during the growing season

Threshold sensitivity must be tested and reported.

---

## 9. SSL Objective Specifications (Formalized in v2.3)

The three locked pretext tasks for process-centered SSL are:

### SSL-A: Masked Spatiotemporal Reconstruction

**Objective:** Randomly mask space-time blocks in the harmonized optical input and train the model to reconstruct them, following MAE (He et al. 2022) but with spatiotemporal masking.

| Parameter | Specification |
|-----------|---------------|
| Spatial masking ratio | 75% of spatial patches per time step |
| Temporal masking ratio | 20% of time steps fully masked |
| Loss | L1 reconstruction loss on masked patches only |
| Rationale | Foundational representation learning; forces model to understand spatial structure and temporal continuity of EO data |

### SSL-B: Phenology Transition Prediction

**Objective:** Given current patch state + weather history, predict the timing of the next phenological transition (e.g., emergence, flowering, maturity, senescence).

| Parameter | Specification |
|-----------|---------------|
| Input | Current vegetation state + weather + crop calendar prior |
| Target | Days until next phenological stage transition (derived from NDVI/EVI inflection points) |
| Loss | Huber loss on predicted vs observed transition timing |
| Rationale | Forces model to learn crop-specific developmental trajectories; directly supports Phenology Forecast Head |

### SSL-C: Drought-Stress Trajectory Prediction

**Objective:** Given recent rainfall/NDVI/soil moisture, predict near-future vegetation stress trajectory (NDVI(t+delta), EVI(t+delta), soil moisture(t+delta)).

| Parameter | Specification |
|-----------|---------------|
| Input | Recent multi-modal state (weather + satellite + soil moisture) |
| Target | NDVI, EVI, and soil moisture anomalies at t+1, t+2, t+3 (5-day steps ahead) |
| Loss | Multi-step MSE with exponential time discounting (near-term predictions weighted higher) |
| Rationale | Forces model to learn drought evolution dynamics; directly supports Drought Trajectory Head and Crop Stress Head |

**Collapse Monitoring:** Representation collapse is monitored via eigenvalue analysis of the embedding covariance matrix. If effective rank drops below threshold (top-1 eigenvalue explains > 90% of variance), trigger alert and increase temperature/add noise.

---

## 10. Physics Regularization

### 10.1 Water Balance Constraint (Clarified Residual Formulation)

The water-balance residual is:

```
residual = Delta_SM_pred - (P + I_obs + I_latent - ET - R - D)
```

| Variable | Definition | Units | Source |
|----------|-----------|-------|--------|
| Delta_SM_pred | Predicted change in soil moisture | mm/5-day | Model prediction |
| P | Precipitation | mm/5-day | CHIRPS/ERA5 (observed) |
| I_obs | Observed irrigation (where available) | mm/5-day | Irrigation maps (when present) |
| I_latent | Latent residual water input | mm/5-day | Model inference |
| ET | Evapotranspiration | mm/5-day | ERA5 ET0 x crop coefficient |
| R | Surface runoff | mm/5-day | Modeled (SCS-CN or empirical) |
| D | Deep drainage | mm/5-day | Modeled (simplified) |

**Physics loss:**

```
L_water = w_data x ||Delta_SM_pred - (P + I_obs + I_latent - ET - R - D)||^2
```

The residual represents the unexplained water-storage change after accounting for all known and inferred fluxes. Minimizing this residual enforces approximate conservation of mass.

### 10.2 AGDD Growth Constraint

```
GDD(t) = max(0, T_mean(t) - T_base)
AGDD(t) = sum(GDD(s)) for s = planting_date to t
```

| Parameter | Wheat | Maize |
|-----------|-------|-------|
| T_base | 0 deg C | 10 deg C |
| T_upper | 30 deg C | 30 deg C |

**Mitscherlich-Baule growth constraint (simplified):**

```
G(AGDD, SM) = G_max x (1 - exp(-c1 x AGDD)) x (1 - exp(-c2 x SM))
```

### 10.3 Dynamic Lambda Scheduling

```
lambda_physics(epoch) = lambda_max x min(1, epoch / warmup_epochs)
```

| Parameter | Specification |
|-----------|---------------|
| lambda_max | Maximum physics loss weight (sweep range: 0.01-1.0) |
| warmup_epochs | Physics ramp-up period (default: 10) |
| Uncertainty-weighted variant | lambda_physics(t) = 1 / (2 x sigma_physics(t)^2) |

---

## 11. Missing Modality Protocol

### 11.1 Design

For each modality:
1. Modality-present mask (binary indicator per sample per modality)
2. Learned missing-modality token replaces absent modality tokens
3. Modality dropout during training (p=0.1-0.2)
4. Performance under missing-modality scenarios reported

### 11.2 Missing-Modality Test Matrix

| Test | Description | Acceptance Criterion |
|------|-------------|---------------------|
| Full modality | All inputs available | Baseline performance |
| No soil moisture | Remove SMAP/ESA CCI | <5% relative drop |
| No irrigation maps | Remove irrigation modality | <3% relative drop (I_latent compensates) |
| No SAR | Optical-only fallback | <10% relative drop |
| Cloud-degraded optical | Mask random time steps | Graceful degradation |
| Weather-only | No satellite imagery | Documented baseline |

---

## 12. Baseline Specifications (Expanded in v2.3)

All baselines evaluated on the same splits with the same statistical protocol.

| # | Baseline | Type | Description | Citation |
|---|----------|------|-------------|----------|
| 1 | Climatology / Historical Mean | Minimum skill | Average conditions over training period | — |
| 2 | Persistence Model | Temporal | Predict current state continues | — |
| 3 | GDD Phenology Model | Simple crop-physics | Accumulate GDD; predict phenology from thermal thresholds | — |
| 4 | SPI/SPEI/VHI Drought Indices | Drought-monitoring | Standard drought indices from climate data | — |
| 5 | Random Forest / XGBoost | Classical ML | Hand-crafted features: NDVI trajectory, weather stats, soil, calendar | — |
| 6 | LSTM / TCN | Deep temporal | Temporal deep learning on time-series features | — |
| 7 | U-Net | Vision | Spatial segmentation for stress detection | Ronneberger et al. 2015 |
| 8 | Temporal ViT | Vision-temporal | ViT with temporal attention (non-process SSL) | — |
| 9 | Single-modality (optical only) | Ablation | Full model, satellite-only input | — |
| 10 | Single-modality (weather only) | Ablation | Full model, weather-only input | — |
| 11 | Single-task (drought only) | Ablation | Full model, drought objective only | — |
| 12 | Single-task (phenology only) | Ablation | Full model, phenology objective only | — |
| 13 | Generic MAE SSL | Ablation | Full model with generic MAE (He et al. 2022) instead of process SSL | He et al. 2022 |
| 14 | Unconstrained (no physics) | Ablation | Full model without water-balance/growth constraints | — |
| 15 | Non-causal (no DAG) | Ablation | Full model without temporal DAG constraints | — |
| 16 | Prithvi-EO-2.0 | GeoFM | Pre-trained geospatial foundation model (IBM/NASA) | Jakubik et al. 2024 |
| 17 | DOFA | GeoFM | Dynamic One-For-All model for multi-sensor EO | Xiong et al. 2024 |

**Note on GeoFM baselines (#16, #17):** These are included if feasible within compute budget. If pre-trained checkpoints are available and can be fine-tuned on our tasks, they provide the strongest comparison. If not feasible, report as "not evaluated due to compute constraints."

---

## 13. Uncertainty Quantification Metrics (NEW in v2.3)

The Residual Water/Uncertainty Head produces predictive uncertainty estimates. These must be calibrated and evaluated:

| Metric | Full Name | Purpose | Formula / Description | MVP Target |
|--------|-----------|---------|----------------------|------------|
| ECE | Expected Calibration Error | Measures reliability of predicted probabilities | Weighted average of absolute difference between predicted confidence and observed accuracy across bins | ECE < 0.10 |
| CRPS | Continuous Ranked Probability Score | Proper scoring rule for probabilistic forecasts | Integral of squared difference between predicted CDF and observed CDF | Competitive with ensemble baselines |
| PICP | Prediction Interval Coverage Probability | Coverage of prediction intervals | Fraction of observations falling within predicted 95% confidence interval | 90-95% coverage for well-calibrated intervals |

---

## 14. Code Style Guidelines

- **Standard:** PEP 8.
- **Type Hints:** Required for all function and class signatures.
- **Docstrings:** NumPy style detailing inputs, outputs, and mathematical formulas.

---

## 15. Testing Strategy

- **Framework:** PyTest.
- **Coverage Target:** Minimum 85% overall, 100% on physics loss formulas.

### 15.1 Unit Test Boundaries

| Test File | Key Assertions |
|-----------|----------------|
| `test_datacube.py` | Zero-leak (no future in past); modality masks correct; 64x64/128x128 patches aligned; EO harmonization RMSE < 0.02 |
| `test_encoders.py` | Correct output shapes; gradient flows through all paths; harmonized input produces consistent embeddings |
| `test_physics.py` | Water-balance loss = 0.0 under perfect conservation; AGDD matches hand-computed; Mitscherlich-Baule zero at zero thermal/moisture |
| `test_diagnostics.py` | DAG has no cycles; no back-in-time gradients; ECE computed correctly; spatial block bootstrap valid CIs |
| `test_ssl.py` | Collapse detection triggers on degenerate representations; SSL-A/B/C objectives produce correct loss values |

---

## 16. Success Criteria (MVP + Stretch Split, Frozen)

| Criterion | Metric | MVP Target | Stretch Target | Claim |
|-----------|--------|------------|----------------|-------|
| Model feasibility | Multi-resolution dataloader | Correct token shapes | — | — |
| Physics consistency | Water-budget RMSE reduction vs unconstrained | >= 15% reduction | >= 25% reduction | C4 |
| Physics non-harm | Downstream metric decrease | <= 3% decrease | <= 1% decrease | C4 |
| Crop Stress AUROC (in-region) | AUROC | >= 0.90 | >= 0.93 | C1, C2 |
| Crop Stress AUROC (transfer) | AUROC | >= 0.80 | >= 0.85 | C2, C6 |
| Phenology MAE | Days | <= 5.0 | <= 3.0 | C1 |
| Yield Risk R-squared | District yield anomaly | >= 0.50 | >= 0.70 | C1, C2 |
| Yield Risk RMSE | % anomaly | < 15% | < 10% | C1, C2 |
| Causal plausibility | DAG sign consistency | >= 90% consistent with agronomic priors; violations documented and investigated | >= 95% consistent | C5 |
| Process SSL advantage | Transfer/fine-tune efficiency | >= 5% relative improvement over generic MAE on at least 1 primary task or transfer split | >= 10% on 2+ tasks | C3 |
| Extreme-event robustness | Event-level F1 on drought holdouts | >= 10% relative improvement over LSTM/XGBoost | >= 20% improvement | C6 |
| Joint modeling benefit | Joint vs single-task | p < 0.05 on at least 2 of 3 primary tasks | p < 0.01 on all 3 tasks | C1 |
| I_latent interpretability | Correlation with irrigation areas | p < 0.05 | p < 0.01 | C4 |
| Uncertainty calibration | ECE | < 0.10 | < 0.05 | — |
| Drought trajectory | RMSE vs persistence | >= 10% relative reduction | >= 20% reduction | — |
| Missing modality robustness | Performance drop | <5% for non-critical; <10% for any single modality | <3% for non-critical; <5% for any | C2 |

---

## 17. Cloud Contamination Protocol

| Scenario | Strategy |
|----------|----------|
| 5-day composite <20% cloud | Use optical directly |
| 20-80% cloud | Flag as partially contaminated; use with quality weight |
| >80% cloud | Fall back to SAR/HLS/monthly; flag as gap-filled |
| Persistent cloud (>3 consecutive) | Monthly aggregation; report as cloud-gap event |
| Monsoon regions | SAR as primary; optical secondary |

---

## 18. Storage and Compute Estimates

### 18.1 Storage Estimates (4 regions, 26 years, sparse patches)

| Category | Per Region | 4 Regions Total |
|----------|-----------|-----------------|
| Harmonized optical (HLS + pre-HLS, sparse patches) | ~300-500 GB | ~1.2-2.0 TB |
| ERA5/CHIRPS | ~60 GB | ~240 GB |
| SoilGrids (static) | ~5 GB | ~20 GB |
| SMAP/ESA CCI | ~20 GB | ~80 GB |
| Crop/irrigation maps | ~10 GB | ~40 GB |
| Yield labels | ~1 GB | ~4 GB |
| Processed datacubes | ~100-200 GB | ~400-800 GB |
| **Total** | **~500-800 GB** | **~2.0-3.2 TB** |

### 18.2 Compute Estimates

| Task | GPU-Hours | Hardware |
|------|-----------|----------|
| Datacube preprocessing | 200-500 CPU-hours | CPU |
| Baselines (RF/XGBoost) | 50-100 CPU-hours | CPU |
| Baselines (LSTM/TCN/U-Net/Temporal ViT) | 200-500 GPU-hours | 1x A100 |
| SSL pretraining (SSL-A/B/C) | 2,000-5,000 GPU-hours | 4-8x A100 |
| Full model training | 2,000-4,000 GPU-hours | 8x A100 |
| Physics experiments | 500-1,000 GPU-hours | 4-8x A100 |
| Ablations (15+ baselines + patch size) | 2,000-4,000 GPU-hours | 4-8x A100 |
| Transfer/robustness/bootstrap | 500-1,000 GPU-hours | 4-8x A100 |
| GeoFM baselines (Prithvi/DOFA fine-tuning) | 500-1,000 GPU-hours | 4-8x A100 |
| **Total** | **8,000-16,000 GPU-hours** | **8x A100 (80GB)** |

---

## 19. 52-Week MVRC Roadmap (Frozen)

### Phase 1: Data Engineering & Spatiotemporal Datacube (Weeks 1-14)

| Sprint | Weeks | Tasks | Gate |
|--------|-------|-------|------|
| 1.1 | 1-3 | Ingest ERA5/CHIRPS (2000-2026) | — |
| 1.2 | 4-7 | Fetch Landsat 5/7/8 + Sentinel-2; implement EO harmonization; fetch irrigation maps | G0: Scope Lock |
| 1.3 | 8-11 | Extract 1,000-5,000 representative 64x64 + 128x128 patches; mask clouds | — |
| 1.4 | 12-14 | Multi-resolution dataloader; 5-fold blocked split indexes; zero-leak verification | G1: Data Readiness |

### Phase 2: Core Architecture & Weak Physics (Weeks 15-28)

| Sprint | Weeks | Tasks | Gate |
|--------|-------|-------|------|
| 2.1 | 15-18 | Implement all encoders + EO harmonization integration; AGDD tracking | G2: Baseline Performance |
| 2.2 | 19-21 | Scale-aware fusion; hierarchical temporal memory | — |
| 2.3 | 22-24 | Temporal DAG (structural prior); gradient audit | G3: Multimodal Fusion Value |
| 2.4 | 25-28 | Water-balance loss; AGDD growth constraint; dynamic lambda scheduling | — |

### Phase 3: Model Training & Supervised Heads (Weeks 29-42)

| Sprint | Weeks | Tasks | Gate |
|--------|-------|-------|------|
| 3.1 | 29-33 | Train SSL-A (reconstruction), SSL-B (phenology), SSL-C (stress trajectory); monitor collapse | G4: Process SSL Value |
| 3.2 | 34-38 | Attach 3 primary + 2 auxiliary heads; train with joint loss; uncertainty quantification | — |
| 3.3 | 39-42 | Hyperparameter sweeps; patch-size ablation (64x64 vs 128x128) | G5: Physics Value |

### Phase 4: Rigorous Validation & Claims Audit (Weeks 43-52)

| Sprint | Weeks | Tasks | Gate |
|--------|-------|-------|------|
| 4.1 | 43-46 | Spatial block bootstrap; year-level LOO; ECE/CRPS calibration | — |
| 4.2 | 47-49 | Extreme drought holdouts; geographic/sensor/crop OOD evaluation | G6: Causal-Informed Value |
| 4.3 | 50-51 | Physics/causal consistency audits | — |
| 4.4 | 52 | Baseline comparison finalization; ablation tables; error analysis; manuscript results | G7: Research Readiness |

---

## 20. Implementation Gates

| Gate | Name | Go | No-Go | Decision |
|------|------|----|-------|----------|
| G0 | Scope Lock | Final selection testable and data exists | No data for any region-crop | Redefine scope |
| G1 | Data Readiness | >=2 regions sufficient; missingness acceptable | No reliable labels | Fix pipeline |
| G2 | Baseline Performance | >=1 baseline beats climatology; eval pipeline stable | Labels invalid | Revisit task |
| G3 | Multimodal Fusion Value | Fusion improves over best single-modality | No fusion benefit | Analyze attention |
| G4 | Process SSL Value | Process SSL improves over generic MAE | SSL collapses | Revert to generic MAE |
| G5 | Physics Value | >=15% residual reduction, <=3% task drop | Physics worsens both | Report as diagnostic |
| G6 | Causal-Informed Value | DAG passes sign/monotonicity checks | Unstable scenario behavior | Interpretability only |
| G7 | Research Readiness | Ablations support >=1 contribution; reproducible | Inconclusive | Extend analysis |

---

## 21. Risk Register

| ID | Risk | Severity | Mitigation | Fallback |
|----|------|----------|------------|----------|
| R1 | Overbroad scope | High | MVRC for 52 weeks | Narrow scope |
| R2 | Pre-Sentinel data gaps | High | EO harmonization; modality masks | Restrict to 2017+ |
| R3 | SSL collapse | High | Collapse monitoring; temperature scheduling | Revert to generic MAE |
| R4 | Physics destabilizes training | Medium | Dynamic lambda with warmup | Remove physics |
| R5 | Irrigation data gaps | High | I_latent; modality masks; coverage reporting | Treat as confounder |
| R6 | Phenology pseudo-label noise | Medium | Noise estimation; soft labels | Report as exploratory |
| R7 | Compute exceeded | Medium | Prioritize C1-C3; reduce patches | Use 1,000 patches |
| R8 | I_latent uninterpretable | Medium | Validate against irrigation areas | Remove I_latent |
| R9 | Cross-region transfer fails | Medium | Check alignment, attention | Report negative transfer |
| R10 | Causal DAG fails invariance | Low | Treat as interpretability only | Report as analysis |
| R11 | EO harmonization insufficient | Medium | Validate on overlap period; report RMSE | Restrict to Sentinel era |
| R12 | 128x128 patch ablation shows major gap | Low | Report as sensitivity finding | Use 128x128 for paper |

---

## 22. Terminology Compliance

| Allowed | Prohibited (in MVRC) | Reason |
|---------|---------------------|--------|
| Crop-climate risk | Food security risk | Requires socioeconomic data |
| Yield anomaly / yield shortfall | Food shortage | Overreaches scope |
| Crop stress detection | Famine prediction | Not supported |
| Agricultural production risk | Food crisis | Overreaches |
| I_latent (latent residual water input) | Irrigation detection | Captures multiple unobserved water sources |
| Process-centered SSL | General self-supervised learning | Must specify process-centered nature |
| Weak physics regularization | Physics-informed model | "Weak" acknowledges incompleteness |
| Temporal DAG constraint (structural prior) | Causal discovery | Structural prior, not full discovery |
| High directional consistency | 100% sign consistency | Biological systems are non-monotonic |

---

## 23. Key References

| Ref | Citation | Relevance |
|-----|----------|-----------|
| He et al. 2022 | "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022 | SSL-A foundation; MAE baseline |
| Chen et al. 2020 | "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020 (SimCLR) | Contrastive SSL context |
| Brenning 2012 | "Spatial cross-validation and bootstrap for the assessment of prediction rules," GMD | Justifies spatial block CV and bootstrap |
| Karasiak et al. 2021 | "Spatial cross-validation for satellite image classification," IGARSS | Justifies spatial CV for EO |
| Claverie et al. 2018 | "The Harmonized Landsat and Sentinel-2 surface reflectance data set," Remote Sensing of Environment | HLS product for EO harmonization |
| Chastain et al. 2019 | "An evaluation of the consistency of Landsat 8 and Sentinel-2 data," Remote Sensing | Cross-sensor calibration |
| Van Tricht et al. 2023 | "WorldCereal: A global dynamic seasonal cropland and irrigation map," IJDE | WorldCereal irrigation maps |
| Funk et al. 2015 | "The climate hazards infrared precipitation with stations (CHIRPS)," Scientific Data | CHIRPS rainfall dataset |
| Jakubik et al. 2024 | "Prithvi-EO-2.0," IBM/NASA | GeoFM baseline |
| Xiong et al. 2024 | "DOFA: Dynamic One-For-All," CVPR | GeoFM baseline |
| Liu et al. 2025 | Nature Communications | 2012 US drought caused ~30% maize yield loss |
| Raissi et al. 2019 | "Physics-informed neural networks," JCP | PINNs for physics regularization |
| Pearl 2009 | "Causality: Models, Reasoning, and Inference," Cambridge | Causal framework context |

---

## 24. Validation Summary

This v2.3 specification has been validated through:

1. **Internal review (v2.2):** 4 critical issues, 6 high-priority, 8 medium, 7 low — all resolved.
2. **External review round 1** (scored 9.0-9.5 novelty/rigor): 10 mandatory revisions identified.
3. **External review round 2** (comprehensive feasibility audit): Confirmed same 10 revisions plus citation requirements.

All mandatory revisions are incorporated in this v2.3 document:

1. EO harmonization layer (HLS preferred, cross-sensor calibration for pre-2015)
2. Irrigation as preferred covariate (not required) with coverage reporting
3. DAG sign consistency changed from "100%" to ">=90% consistent with agronomic priors"
4. Success criteria split into MVP targets and stretch goals
5. Explicit 4-dimensional OOD definitions (geographic, temporal/extreme, sensor, crop/region)
6. Three locked SSL pretext tasks (SSL-A/B/C) formally specified
7. Water-balance residual clearly formulated over P + I_obs + I_latent - ET - R - D
8. Yield MVP target lowered to R-squared >= 0.50 (stretch >= 0.70)
9. Baseline suite expanded to 17 baselines including Prithvi-EO-2.0, DOFA, U-Net, Temporal ViT
10. Uncertainty quantification metrics (ECE, CRPS, PICP) added
11. Patch-size ablation (64x64 baseline, 128x128 ablation) added
12. Paper reframed as "climate-resilience foundation model"
13. Causal DAG clarified as "structural prior, not full causal discovery"
14. Key citations added (Brenning 2012, He 2022, Chen 2020, Claverie 2018, etc.)

**Status: FROZEN — Ready for implementation.**
