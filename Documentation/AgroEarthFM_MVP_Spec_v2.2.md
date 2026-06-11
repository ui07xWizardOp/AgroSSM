# AgroEarthFM-MVP Research Prototype Specification v2.2

**Validated & Finalized — Incorporating All Review Corrections**

**Document Version:** 2.2 (Finalized)
**Date:** 2026-05-29
**Status:** Pre-Implementation Specification — Validated
**Classification:** Research Prototype Specification
**Supersedes:** v2.1 (2026-05-22)

---

## Revision Log

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | 2026-05-22 | Complete Finalized Specification from Scope-Lock session |
| v2.1 | 2026-05-22 | Draft incorporating grill-me session changes |
| v2.2 | 2026-05-29 | **Comprehensive review & validation**: Fixed irrigation policy contradiction (C1), resolved 2022 IGP data-split conflict (C2), added Data Availability by Era table for 2000–2026 expansion (C3), added missing success criteria for C3 and C6 (C4), added Sprint 4.4 to timeline (H1), added Po Valley data-availability table (H2), added missing project structure modules (H3), clarified I_latent role (H4), updated storage/compute estimates (H5), added cross-platform commands (H6), corrected extreme drought year for Spain (M1), added baseline specifications (M2), added dynamic lambda scheduling (M3), mapped verification commands to test functions (M4), added label noise estimation (M5), added AGDD and growth constraint formulas (M6), integrated implementation gates into phase structure (M7), cleaned up documentation references (M8), added MVRC deliverable definition (L1), carried forward missing-modality test matrix (L2), scale compliance rule (L3), no-leakage protocol (L4), cloud contamination protocol (L5), resolution tiers (L6), risk register (L7). |

---

## 1. Objective

Build a multimodal, spatiotemporally sparse, physics-regularized, and causally-informed machine learning research prototype for crop stress detection, crop phenology forecasting, and agricultural yield-risk/anomaly prediction.

The goal is to deliver a Year-1 Minimum Viable Research Contribution (MVRC) that demonstrates:

1. **C1 (Joint Modeling):** Joint drought + phenology representation improves performance over single-task baselines.
2. **C2 (Multimodal Fusion):** Multi-source input improves out-of-distribution (OOD) robustness.
3. **C3 (Process SSL):** Process-centered Self-Supervised Learning outperforms general masked autoencoders.
4. **C4 (Physics Regularization):** Weak water-balance constraints reduce physical inconsistencies without degrading metric performance.
5. **C5 (Causal Diagnostics):** Structured temporal Directed Acyclic Graph (DAG) constraints provide biologically plausible representations.
6. **C6 (Tail Events):** Robustness under extreme weather events and spatial-temporal block holdouts.

### 1.1 MVRC Deliverable Definition

The Year-1 MVRC consists of three deliverables:

| Deliverable | Description | Completion Criteria |
|-------------|-------------|---------------------|
| **Research Paper** | A manuscript-ready paper with ablation tables, error analysis, and validated claims C1–C6 | At least 1 claim with statistically significant positive result; all claims reported (positive or negative) |
| **Code Repository** | Fully reproducible codebase with frozen dependencies, documented configs, and fixed train/val/test splits | `pytest` passes; any experiment in the paper reproducible from config + splits |
| **Model Checkpoint** | Final trained model checkpoint with evaluation scripts | Checkpoint loads and produces reported metrics within 1% tolerance |

### 1.2 Claim Validation Priority

| Priority | Claims | Justification |
|----------|--------|---------------|
| Must-validate | C1, C2, C3 | Core novelty of the paper |
| Strongly-validate | C4, C6 | Key differentiators |
| Validate-if-possible | C5 | High-risk, high-reward; acceptable as interpretability contribution |

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
- **Data Processing:** Xarray, Rasterio, Pandas, NumPy, Scipy
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
│   └── AgroEarthFM_MVP_Spec_v2.2.md  # This specification document
├── scripts/
│   ├── preprocess_data.py          # Sparsity extraction and datacube builder
│   ├── train_model.py              # Main training script (PyTorch Lightning)
│   └── evaluate_model.py           # Evaluation, bootstrapping, and causal audits
├── src/
│   ├── datacube/
│   │   ├── __init__.py
│   │   ├── datacube_loader.py      # Spatial-temporal patch dataset loading
│   │   ├── patch_extractor.py      # 64x64 representative agricultural patch extractor
│   │   └── modality_masks.py       # Missing-modality token and modality dropout logic
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── satellite_encoder.py    # ViT-based Sentinel-2/Landsat encoder
│   │   ├── weather_encoder.py      # Temporal weather transformer encoder
│   │   ├── soil_encoder.py         # Static MLP encoder for soil/topography
│   │   └── calendar_encoder.py     # Crop stage prior embedding
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── cross_attention.py      # Scale-aware cross-attention layer
│   ├── ssl/
│   │   ├── __init__.py
│   │   ├── masked_reconstruction.py  # SSL-1: Masked spatiotemporal reconstruction
│   │   ├── cross_modal_prediction.py # SSL-2: Cross-modal masked prediction
│   │   ├── trajectory_prediction.py  # SSL-3/5: Future vegetation & drought trajectory
│   │   ├── phenology_prediction.py   # SSL-4: Phenology pseudo-transition prediction
│   │   └── collapse_monitor.py       # Eigenvalue-based representation collapse detection
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
│   │   └── bootstrap.py            # Spatial block bootstrap & Year LOO bootstrap
│   └── baselines/
│       ├── __init__.py
│       ├── climatology.py          # Historical mean baseline
│       ├── persistence.py          # Temporal persistence baseline
│       ├── gdd_phenology.py        # GDD-based phenology model
│       ├── drought_indices.py      # SPI/SPEI/VHI drought index models
│       ├── tree_models.py          # Random Forest / XGBoost baselines
│       └── lstm_tcn.py             # LSTM / TCN temporal deep learning baselines
└── tests/
    ├── __init__.py
    ├── test_datacube.py            # Unit tests for patch loading, missing modalities, zero-leak
    ├── test_encoders.py            # Dimension and tensor flow tests
    ├── test_physics.py             # Validation of water-balance constraints
    ├── test_diagnostics.py         # Autocyclicity and gradient leak tests
    └── test_ssl.py                 # Collapse monitor, masking correctness, objective convergence
```

---

## 6. Architectural Specifications (Resolved from Grill-Me & Review)

### 6.1 Output Heads

The network outputs are divided into:

**Primary Supervised Downstream Heads:**

| Head | Output | Metric | Success Criterion |
|------|--------|--------|-------------------|
| **Crop Stress Head** | Probability of crop stress (water/heat) and spatial IoU | AUROC | >= 0.90 in-region; >= 0.80 in spatial/temporal transfer |
| **Phenology Forecast Head** | Expected timing of next phenological transition | MAE in days | <= 5.0 days |
| **Yield-Risk Head** | District/regional yield anomaly percentage | RMSE / R-squared | RMSE < 10%; R-squared >= 0.70 on in-region tests |

**Auxiliary Diagnostic Heads:**

| Head | Output | Metric | Success Criterion |
|------|--------|--------|-------------------|
| **Drought Trajectory Head** | Short-to-medium term drought progression anomalies | RMSE (SPI/SPEI anomaly) | Must outperform SPI/SPEI persistence baseline by >= 10% relative RMSE reduction |
| **Residual Water/Uncertainty Head** | Predictive uncertainty and I_latent inference | Correlation with known irrigation maps | Statistically significant correlation (p < 0.05) between I_latent and irrigated areas |

### 6.2 Irrigation Input Policy (CORRECTED from v2.1)

**v2.1 Issue:** v2.1 stated "The model REQUIRES irrigation maps as direct inputs. It is restricted to regions where irrigation maps are available." This contradicted the v2.0 locked decision that I_latent handles unobserved irrigation, and irrigation maps are used only as validation references.

**Resolution (v2.2 FINAL):**

Irrigation maps are treated as **available covariates with modality masks**, NOT as required inputs. The policy is:

1. Where irrigation maps exist (ESA WorldCereal, regional registries, IWMI, GMIA), they enter the model as an **input modality** with a modality-present binary mask and learned missing-modality placeholder token.
2. Where irrigation maps are unavailable, the modality-present mask is set to 0 and the learned missing-modality token is used.
3. The model is NOT restricted to regions with irrigation map availability; it can operate on any region in the Crop-Region Matrix.
4. **I_latent** (Latent Residual Water-Input Inference) remains the key mechanism for capturing unobserved water inputs. It is explicitly defined as:

   > I_latent: Residual water input NOT captured by available irrigation maps — includes unreported irrigation, groundwater contribution, capillary rise, and measurement error in known irrigation estimates.

5. I_latent is inferred by the Residual Water/Uncertainty Head and enters the water-balance equation as a latent term.
6. During training, **modality dropout** (probability p=0.1-0.2) randomly zeros out entire modalities (including irrigation maps when available) to ensure the model can function without any single modality.

**Rationale:** Making irrigation maps a required input would (a) limit geographic scope to areas with fine-scale irrigation data, (b) create a dependency on data with known quality issues at fine spatial scales, and (c) undermine the I_latent mechanism which was specifically designed to handle this gap.

### 6.3 Causal Chain and Temporal DAG (FINAL)

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

### 6.4 Scope, Spatiotemporal Sparsity, & Regional Matrix

- **Temporal Coverage:** 2000-2026 (see Section 8 for Data Availability by Era).
- **Crop-Region Matrix (Balanced & De-confounded):**

  | Crop | Region 1 | Region 2 |
  |------|----------|----------|
  | Wheat | Indo-Gangetic Plain (India) | Iberian Peninsula (Spain) |
  | Maize | US Corn Belt (Iowa/Illinois) | Po Valley (Italy) |

- **Sparsity Optimization:** To reduce the 40+ TB storage and 50k+ GPU-hour requirements, training utilizes **Spatiotemporal Sparsity**. The system extracts and trains only on 1,000-5,000 representative 64x64 agricultural patches across each region. The code structure remains flexible (modular interfaces) to swap this with frozen encoders or MODIS coarse-to-fine pre-training.

### 6.5 Data Splits & Temporal Blocked K-Fold (CORRECTED from v2.1)

- **Chronological Split:**
  - **Train/Validation Pool:** 2000-2021
  - **In-Region Test set:** 2023-2025 (see correction below)
  - **Extreme Drought Years (Held out entirely for extreme-event evaluation):** 2005 (Spain), 2012 (US Corn Belt), 2022 (IGP)

**v2.1 Issue:** v2.1 listed the in-region test set as "2022-2025" while also listing 2022 IGP as an extreme drought holdout. This meant 2022 IGP data was in both the normal test set and the extreme holdout.

**Resolution (v2.2 FINAL):** The in-region test set for each crop-region combination EXCLUDES its corresponding extreme drought year. The corrected test sets are:

| Region-Crop | In-Region Test Years | Extreme Drought Holdout |
|-------------|---------------------|------------------------|
| IGP Wheat | 2023-2025 | 2022 |
| US Corn Belt Maize | 2023-2025 | 2012 |
| Iberian Peninsula Wheat | 2023-2025 | 2005 |
| Po Valley Maize | 2023-2025 | (none specifically held out; included in normal test) |

- **Validation Protocol:** Within the 2000-2021 pool, hyperparameters are tuned using **5-Fold Spatial-Temporal Blocked Cross-Validation** to eliminate spatial and temporal leakage.

- **Split locking rule:** All splits must be defined BEFORE model selection and training begins to prevent data leakage. Region/year/event splits are fixed and committed to version control.

### 6.6 Autocorrelation & Statistical Protocol

- Naive pixel-level random splits and t-tests are prohibited.
- Performance statistics (confidence intervals, p-values) are calculated using a **Spatial Block Bootstrap (50km x 50km blocks)** and a **Year-Level Leave-One-Out (LOO) Bootstrap**.

### 6.7 No-Leakage Protocol for Forecasting Tasks

1. Future satellite observations beyond forecast date are PROHIBITED in inputs.
2. Full-season composites in early-season predictions are PROHIBITED.
3. Yield labels or statistics from the target prediction period are PROHIBITED.
4. Gap-filled products that use future observations are PROHIBITED unless explicitly documented.
5. Region-level normalization using held-out region statistics is PROHIBITED.
6. All preprocessing (normalization parameters, cloud thresholds) is fit only on training data.

### 6.8 Scale Compliance Rule

Predictions are validated ONLY at the spatial scale of the ground truth labels. If labels are district-level, predictions are aggregated to district level before computing yield metrics. Never claim field-level yield accuracy from administrative-level labels.

---

## 7. Complete Model Architecture

### 7.1 MVP Architecture (Final)

```
Fine Satellite Encoder (ViT-based)
        |
Weather / Climate Temporal Encoder (Temporal Transformer)
        |
Soil / Static Context Encoder (MLP)
        |
Crop Calendar / Crop Type Embedding (Learned Embedding)
        |
Irrigation Map Encoder (with modality-present mask + missing-modality token)
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
Temporal DAG Constraints (causal inductive bias)
```

### 7.2 Component Specifications

#### 7.2.1 Fine Satellite Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | Vision Transformer (ViT) |
| Input | Sentinel-2 (2017-2026) / Landsat 5/7/8 (2000-2016) multi-spectral imagery |
| Patch size | 16x16 pixels |
| Input bands | B2, B3, B4, B8 (10m) + B5, B6, B7, B8A, B11, B12 (20m resampled) |
| Temporal input | Sequence of 5-day composites |
| Positional encoding | 2D spatial + temporal encoding |
| Output | Token sequence per time step |
| Pretraining | SSL objectives (masked spatiotemporal reconstruction) |
| Approximate parameter count | 30-50M parameters |

#### 7.2.2 Weather / Climate Temporal Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | Temporal Transformer |
| Input | ERA5 / CHIRPS climate variables at native resolution (~31km / ~5km) |
| Variables | Temperature (min/max/mean), precipitation, wind speed, solar radiation, humidity, ET0 |
| Temporal resolution | Daily aggregated to 5-day windows |
| Positional encoding | Temporal encoding with day-of-year and season indicators |
| Output | Climate state vector per time step |
| Approximate parameter count | 10-15M parameters |

#### 7.2.3 Soil / Static Context Encoder

| Parameter | Specification |
|-----------|---------------|
| Base architecture | MLP |
| Input | SoilGrids variables at ~250m resolution |
| Variables | Clay content, sand content, organic carbon, pH, CEC, bulk density, soil depth |
| Output | Static soil-context vector |
| Approximate parameter count | 2-5M parameters |

#### 7.2.4 Crop Calendar / Crop Type Embedding

| Parameter | Specification |
|-----------|---------------|
| Architecture | Learned embedding layer |
| Input | Crop type (one-hot or embedding index), planting date, expected harvest date |
| Source | GEOGLAM crop calendars, crop masks |
| Output | Crop-context vector per pixel/patch |
| Approximate parameter count | <1M parameters |

#### 7.2.5 Irrigation Map Encoder (NEW in v2.2)

| Parameter | Specification |
|-----------|---------------|
| Architecture | MLP with modality-present mask |
| Input | ESA WorldCereal irrigation classification (10m) / IWMI / GMIA where available |
| Missing modality handling | Binary modality-present mask; learned missing-modality placeholder token |
| Modality dropout | p=0.1-0.2 during training (randomly zero out entire irrigation modality) |
| Output | Irrigation-context vector per pixel/patch |
| Approximate parameter count | 1-2M parameters |

#### 7.2.6 Scale-Aware Fusion Module

| Parameter | Specification |
|-----------|---------------|
| Architecture | Cross-attention with resolution-aware positional encoding |
| Input | Token sequences from all encoders |
| Mechanism | Fine-resolution tokens attend to coarse-resolution context |
| Resolution handling | Multi-resolution tokens preserved; no forced upsampling to 10m |
| Missing modality | Modality-present mask + learned missing-modality token |
| Output | Fused multimodal representation per spatial-temporal position |
| Approximate parameter count | 15-25M parameters |

#### 7.2.7 Hierarchical Temporal Memory Module

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
| Satellite Encoder (ViT) | 30-50M | ~256 tokens per time step (16x16 patches in 256x256 input) |
| Weather Encoder | 10-15M | ~1-4 tokens per time step (coarse resolution) |
| Soil Encoder | 2-5M | ~1-4 tokens per time step (static, 250m) |
| Crop Calendar Embedding | <1M | ~1 token per time step |
| Irrigation Map Encoder | 1-2M | ~1-4 tokens per time step |
| Fusion Module | 15-25M | Cross-attention queries: fine tokens; keys/values: all tokens |
| Temporal Memory | 20-30M | 5-day: ~24 tokens (120 days); Monthly: ~6 tokens (6 months) |
| Output Heads | 2-5M | Per-head MLP |
| **Total MVP** | **~81-133M** | **~285-295 tokens per time step** |

**Compute consideration:** This model size is designed to be trainable on a single 8xA100 (80GB) GPU node. The token budget is constrained to ensure batch sizes of at least 4-8 per GPU during SSL pretraining.

---

## 8. Data Strategy

### 8.1 Data Availability by Era (NEW in v2.2)

The expansion from 2017-2023 (v2.0) to 2000-2026 (v2.1+) introduces significant variation in data availability. This table documents what is available in each era:

| Era | Years | Fine Optical | Coarse Optical | SAR | Weather | Soil Moisture | Irrigation Maps | Yield Labels |
|-----|-------|-------------|----------------|-----|---------|---------------|-----------------|--------------|
| **Pre-Sentinel** | 2000-2012 | Landsat 5 TM (30m, 16-day); Landsat 7 ETM+ (SLC-off from May 2003, ~22% data loss per scene) | MODIS (250m-1km, daily) | Not available | ERA5 (31km) | Not available (pre-SMAP) | Not available | Available (USDA, ICRISAT, Eurostat) |
| **Landsat Transition** | 2013-2016 | Landsat 8 OLI (30m, 16-day); Landsat 7 ETM+ (SLC-off) | MODIS (250m-1km, daily) | Sentinel-1 (from 2014, limited coverage) | ERA5 (31km) | SMAP (from 2015, limited) | Not available | Available |
| **Sentinel Era** | 2017-2020 | Sentinel-2 (10m, 5-day) + Landsat 8; HLS (from 2015) | MODIS | Sentinel-1 (full coverage) | ERA5 + ERA5-Land | SMAP (full); ESA CCI | Partial (GMIA, IWMI) | Available |
| **Modern Era** | 2021-2026 | Sentinel-2 + Landsat 8/9 + HLS | MODIS | Sentinel-1 | ERA5 + ERA5-Land | SMAP + ESA CCI | ESA WorldCereal (from 2021) | Available |

**Implications:**
- Pre-2013 data has lower observation density and no SAR; the model must handle this gracefully via missing-modality protocol.
- Landsat 7 SLC-off gaps (2003+) are documented and must be masked in the preprocessing pipeline.
- Irrigation maps are only available in the modern era for high-resolution products (ESA WorldCereal from 2021); coarser products (GMIA, IWMI) provide static estimates applicable across all eras.
- The spatiotemporal sparsity approach (1,000-5,000 patches) mitigates the storage impact of the extended temporal range.

### 8.2 Multi-Resolution Datacube Design

**Core Principle:** The model does NOT force all data to a common 10m grid. Each variable is represented at its appropriate spatial and temporal scale. The fusion module handles multi-resolution inputs natively.

**Resolution Tiers:**

| Tier | Spatial Scale | Temporal Scale | Data Types | Token Treatment |
|------|---------------|----------------|------------|-----------------|
| Fine | 10-30m | 5-day composites | Sentinel-2/Landsat reflectance, vegetation indices | Fine-resolution visual/vegetation tokens |
| Intermediate | 100-500m | 5-day to monthly | Aggregated vegetation dynamics, SAR-derived moisture proxies | Aggregated context tokens |
| Coarse | 1-10km | Daily to monthly | ERA5/CHIRPS weather, SMAP/ESA CCI soil moisture, drought indices | Coarse temporal weather/moisture tokens |
| Regional | Admin/field level | Seasonal/annual | Yield statistics, management records, crop maps | Regional context tokens |

**Datacube Construction Protocol:**

1. **Coregistration:** All data within each tier reprojected to common CRS (EPSG:4326 or UTM zone-specific)
2. **Temporal alignment:** All data aligned to 5-day composite windows (Day 1-5, 6-10, ..., 361-365)
3. **Cloud handling:** 5-day composites use median compositing with valid-pixel filtering; cloud threshold documented per region/season
4. **No cross-tier resampling:** Fine-tier data NOT upsampled from coarse-tier; coarse-tier data NOT forced to 10m
5. **Metadata preservation:** Each variable retains its native resolution, uncertainty estimate, and source metadata
6. **Normalization:** Per-variable normalization using training-period statistics only

### 8.3 Complete Data Source Catalog

#### Satellite Imagery

| Source | Product | Resolution | Bands | Access | License |
|--------|---------|------------|-------|--------|---------|
| Sentinel-2 | L2A reflectance | 10-20m | B2-B12, B8A | Copernicus Open Access Hub / GEE | CC-BY-SA 4.0 / Free |
| HLS (Harmonized Landsat Sentinel-2) | HLSL30/HLSS30 | 30m | Visual + NIR + SWIR | LP DAAC / GEE | Public domain |
| Landsat 5 TM | L1/L2 reflectance | 30m | Multi-spectral (7 bands) | USGS / GEE | Public domain |
| Landsat 7 ETM+ | L1/L2 reflectance | 30m | Multi-spectral (8 bands, SLC-off after May 2003) | USGS / GEE | Public domain |
| Landsat 8/9 OLI | L2 reflectance | 30m | Multi-spectral (11 bands) | USGS / GEE | Public domain |

#### Vegetation Indices

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (B8 - B4) / (B8 + B4) | Green vegetation density |
| EVI | 2.5 x (B8 - B4) / (B8 + 6xB4 - 7.5xB2 + 1) | Corrected vegetation density |
| NDWI | (B8 - B11) / (B8 + B11) | Vegetation water content |
| NDMI | (B8 - B11) / (B8 + B11) | Normalized difference moisture index |
| LAI | From PROSAIL inversion or empirical | Leaf area index |
| FAPAR | From PROSAIL inversion or empirical | Absorbed photosynthetically active radiation |

**Note on pseudo-labels:** Phenology transitions derived from vegetation-index derivatives are noisy pseudo-labels. Label noise must be estimated and reported (see Section 8.7).

#### Climate Data

| Source | Variables | Spatial Resolution | Temporal Resolution | Access | License |
|--------|-----------|-------------------|---------------------|--------|---------|
| ERA5 (ECMWF) | T_min, T_max, T_mean, P, wind, RH, solar radiation, ET0 | ~31km | Hourly -> daily -> 5-day | CDS API / GEE | CC-BY-4.0 |
| ERA5-Land | Enhanced land variables | ~9km | Hourly | CDS API | CC-BY-4.0 |
| CHIRPS | Precipitation | ~5km | Daily | CHIRPS API / GEE | Public domain |

#### Soil Data

| Source | Variables | Resolution | Type | Access |
|--------|-----------|------------|------|--------|
| SoilGrids 250m | Clay %, Sand %, Organic C, pH, CEC, Bulk density, Depth | ~250m | Static | ISRIC / GEE |

#### Soil Moisture

| Source | Product | Resolution | Depth | Access |
|--------|---------|------------|-------|--------|
| SMAP | L3 Enhanced | ~9km (enhanced to ~3km) | 0-5cm | NASA DAAC / GEE |
| ESA CCI Soil Moisture | Active + Passive merged | ~25km | 0-5cm | ESA CCI / GEE |

**Critical note:** Soil moisture products provide coarse-resolution surface (0-5cm) estimates. They represent large-area moisture conditions, not field-scale root-zone soil moisture. They should be treated as coarse context, not fine-grained truth.

#### Crop Calendar and Crop Mask

| Source | Variables | Resolution | Access |
|--------|-----------|------------|--------|
| GEOGLAM Crop Calendar | Planting/harvest dates by crop and region | Regional/country | GEOGLAM / USDA |
| ESA WorldCereal | Crop type maps + irrigation classification | 10m | ESA |
| CDL (US) | Crop type | 30m | USDA NASS |
| ISRO Crop Maps (India) | Crop type, kharif/rabi | State/district | Government of India / ISRO |

#### Yield Labels

| Source | Scale | Access | Quality |
|--------|-------|--------|---------|
| USDA NASS (US) | County-level | Public | High quality, annually updated |
| ICRISAT (India) | District-level | Public | Good quality, historical series |
| Eurostat/CAP (EU) | NUTS2/Province | Public | Good quality for EU member states |
| ISTAT (Italy) | Province/Region | Public (with registration) | Good quality; Po Valley coverage |
| FAOSTAT | Country-level | Public | Coarse, useful for cross-country comparison |

#### Irrigation and Management Proxies

| Source | Data Type | Resolution | Access | Era Availability |
|--------|-----------|------------|--------|-----------------|
| GMIA (FAO) | Irrigated area fraction | ~10km | FAO | All eras (static estimate) |
| IWMI Global Irrigation Map | Irrigated/rainfed classification | ~500m | IWMI | All eras (static estimate) |
| ESA WorldCereal | Irrigation classification | 10m | ESA | Modern era only (2021+) |
| USDA FRIS (US) | Irrigation practices by state | Admin | USDA | Most eras |
| Regional Irrigation Registries (Po Valley) | Irrigation district boundaries, water allocation | Admin | Regione Lombardia/Emilia-Romagna (academic API request) | Most eras |

### 8.4 Study Region Selection — Data Availability Verification

#### IGP (Punjab/Haryana, India) — Wheat

| Criterion | Status | Details |
|-----------|--------|---------|
| Crop map available | Yes | ISRO/Gov India crop maps |
| Sufficient cloud-free observations | Yes (rabi season Oct-Mar) | High observation density in rabi season |
| Weather data for all years | Yes | ERA5/CHIRPS |
| Yield labels available | Yes | ICRISAT district-level |
| Known drought events | Yes | 2002, 2009, 2014, 2022 |
| Irrigation information | Yes | GMIA/IWMI (static) |
| 5+ years of data | Yes | Full 2000-2026 coverage |

#### US Corn Belt (Iowa/Illinois) — Maize

| Criterion | Status | Details |
|-----------|--------|---------|
| Crop map available | Yes | CDL 30m |
| Sufficient cloud-free observations | Yes (summer) | Low cloud contamination |
| Weather data for all years | Yes | ERA5/NOAA |
| Yield labels available | Yes | USDA NASS county-level |
| Known drought events | Yes | 2012, 2023 |
| Irrigation information | Yes | USDA FRIS |
| 5+ years of data | Yes | Full 2000-2026 coverage |

#### Iberian Peninsula (Castilla-y-Leon/Andalucia, Spain) — Wheat

| Criterion | Status | Details |
|-----------|--------|---------|
| Crop map available | Yes | ESA WorldCereal (2021+); SIGPAC historic |
| Sufficient cloud-free observations | Yes (spring) | Good observation density in wheat season |
| Weather data for all years | Yes | ERA5 |
| Yield labels available | Yes | Eurostat/CAP NUTS2 |
| Known drought events | Yes | 2005, 2012, 2017, 2019 |
| Irrigation information | Yes | GMIA/IWMI |
| 5+ years of data | Yes | Full 2000-2026 coverage |

#### Po Valley (Lombardia/Emilia-Romagna, Italy) — Maize (NEW in v2.1, Verified in v2.2)

| Criterion | Status | Details |
|-----------|--------|---------|
| Crop map available | Yes | EU LPIS (Land Parcel Identification System) via AGEA; ESA WorldCereal (2021+) |
| Sufficient cloud-free observations | Partial (summer) | Moderate cloud contamination in Po Valley summer; Landsat + Sentinel-2 combined provides adequate revisit |
| Weather data for all years | Yes | ERA5/ARPAE regional weather network |
| Yield labels available | Yes | ISTAT province/regional level; EU CAP declarations |
| Known drought events | Yes | 2003 (European heat wave, significant Po Valley impact), 2017, 2022 |
| Irrigation information | Yes | Consorzio di Bonifica irrigation district maps; Regione Lombardia/Emilia-Romagna agricultural registries |
| 5+ years of data | Yes | Full 2000-2026 coverage |

### 8.5 Phenology Pseudo-label Quality & Noise Estimation

Phenology transitions are derived from NDVI/EVI time-series derivatives (inflection points, local maxima/minima). These are noisy estimates, not ground truth.

**Required noise estimation:**
1. Compare phenology transition dates derived from different indices (NDVI vs EVI vs NDWI) for the same pixel-year; report the standard deviation as a noise estimate.
2. Where available, compare with ground-based phenology observations (e.g., Pan European Phenology Network for Spain/Italy; USDA phenology data for US).
3. Report label noise statistics (mean, std, per-crop per-region) in the Data Readiness Report.
4. During training, use a soft label approach where phenology transition dates have an associated uncertainty rather than treating them as hard labels.

### 8.6 Crop Stress Labels

Crop stress labels are defined as vegetation health anomalies below a threshold:
- NDVI anomaly < -1 standard deviation, OR
- VHI < 35, OR
- Documented drought event during the growing season

These are event-based labels validated against documented drought events. The threshold sensitivity must be tested and reported.

### 8.7 Data Licensing & Access Checklist

| Source | License | Access Method | Restrictions |
|--------|---------|---------------|--------------|
| Sentinel-2 / Landsat | Open / Public domain | Copernicus/USGS/GEE | Academic-friendly |
| ERA5 / CHIRPS | CC-BY-4.0 / Public | CDS API / GEE | Citation required |
| SoilGrids | Open | ISRIC / GEE | Citation required |
| SMAP / ESA CCI | Open | NASA DAAC / ESA CCI / GEE | Citation required |
| ESA WorldCereal | Free for academic use | ESA | Registration required |
| ISRO Crop Maps | Government data | ISRO portal / NRSC | Standard academic registration |
| ISTAT (Italy) | Open with registration | ISTAT API | Registration required |
| ARPAE (Italy) | Open with registration | ARPAE portal | Academic API request |
| Consorzio di Bonifica (Italy) | Varies | Direct request | May require formal academic request |

---

## 9. SSL Objective Specifications

### 9.1 SSL Objective Sequence

Objectives are introduced progressively, NOT trained simultaneously from the start:

| Stage | Objective ID | Objective | Purpose | Activation |
|-------|-------------|-----------|---------|------------|
| SSL-1 | L_ssl_1 | Masked Spatiotemporal Reconstruction | Learn basic EO representations | Phase 2 (Warm-up) |
| SSL-2 | L_ssl_2 | Cross-Modal Masked Prediction | Learn relationships among weather, soil, and vegetation | Phase 2 (Fusion) |
| SSL-3 | L_ssl_3 | Future Vegetation Trajectory Prediction | Learn temporal ecosystem dynamics | Phase 2-3 |
| SSL-4 | L_ssl_4 | Phenology Pseudo-Transition Prediction | Learn crop-stage timing | Phase 3 (Process) |
| SSL-5 | L_ssl_5 | Drought Stress Trajectory Prediction | Learn drought response dynamics | Phase 3 (Process) |

### 9.2 Collapse Monitoring

Representation collapse is monitored via eigenvalue analysis of the embedding covariance matrix:
- Compute the effective rank of the representation (ratio of top-k eigenvalues to sum of all eigenvalues)
- If effective rank drops below a threshold (e.g., top-1 eigenvalue explains > 90% of variance), trigger alert
- Mitigation: increase temperature in contrastive objectives, add noise, reduce model capacity

---

## 10. Physics Regularization

### 10.1 Water Balance Constraint

The soil water balance equation is:

```
Delta_SM = P + I_latent - ET - R - D + epsilon
```

| Variable | Definition | Units | Source |
|----------|-----------|-------|--------|
| Delta_SM | Change in soil moisture | mm/5-day | Predicted by model |
| P | Precipitation | mm/5-day | CHIRPS/ERA5 (observed) |
| I_latent | Latent residual water input (unobserved irrigation + groundwater + capillary rise + measurement error) | mm/5-day | Inferred by model |
| ET | Evapotranspiration | mm/5-day | ERA5 ET0 x crop coefficient (partially observed, partially modeled) |
| R | Surface runoff | mm/5-day | Modeled (SCS-CN or empirical) |
| D | Deep drainage | mm/5-day | Modeled (simplified) |
| epsilon | Residual error | mm/5-day | Unaccounted fluxes |

**Water balance loss:**

```
L_water = w_data x ||Delta_SM_pred - (P + I_latent - ET - R - D)||^2
```

The model predicts Delta_SM_pred and I_latent jointly. The physics loss measures consistency between the predicted soil moisture change and the water-balance equation.

**I_latent acceptance criteria:**
- I_latent values are higher in known irrigated areas (statistically significant difference, t-test p < 0.05)
- I_latent shows seasonal patterns consistent with irrigation timing
- I_latent is NOT simply correlated with precipitation (partial correlation controlling for P < 0.3)

### 10.2 AGDD Growth Constraint

Accumulated Growing Degree Days (AGDD) govern phenology progression:

```
GDD(t) = max(0, T_mean(t) - T_base)
AGDD(t) = sum(GDD(s)) for s = planting_date to t
```

| Parameter | Wheat | Maize |
|-----------|-------|-------|
| T_base | 0 deg C | 10 deg C |
| T_upper | 30 deg C | 30 deg C |

**Phenology stage constraint:** Crop stage transitions must be consistent with AGDD accumulation. A transition from stage s to stage s+1 should not occur when AGDD is far below the historical minimum for that transition.

**Mitscherlich-Baule growth constraint (simplified):**

```
G(AGDD, SM) = G_max x (1 - exp(-c1 x AGDD)) x (1 - exp(-c2 x SM))
```

Where G is biomass accumulation, G_max is maximum biomass, and c1, c2 are crop-specific growth parameters. This constraint ensures that:
- Growth increases with thermal time (diminishing returns)
- Growth increases with soil moisture (diminishing returns)
- Zero growth under zero thermal time or zero soil moisture

### 10.3 Dynamic Lambda Scheduling for Physics Loss Weights

Physics loss weights are NOT fixed hyperparameters. They follow a dynamic schedule:

```
lambda_physics(epoch) = lambda_max x min(1, epoch / warmup_epochs)
```

| Parameter | Specification |
|-----------|---------------|
| lambda_max | Maximum physics loss weight (tuned via hyperparameter sweep, range: 0.01-1.0) |
| warmup_epochs | Number of epochs over which physics weight ramps up (default: 10) |
| Rationale | Prevents physics from destabilizing early training when representations are poor |

Additionally, an uncertainty-weighted variant can be used:

```
lambda_physics(t) = 1 / (2 x sigma_physics(t)^2)
```

Where sigma_physics is a learned parameter representing the model's uncertainty about the physics constraint.

---

## 11. Missing Modality Protocol

### 11.1 Design

For each modality:
1. Add a **modality-present mask** (binary indicator per sample per modality)
2. Add a **learned missing-modality token** that replaces absent modality tokens
3. Use **modality dropout** during training (randomly zero out entire modalities with probability p=0.1-0.2)
4. Report performance under missing-modality scenarios

### 11.2 Missing-Modality Test Matrix

| Test | Description | Acceptance Criterion |
|------|-------------|---------------------|
| Full modality | All inputs available | Baseline performance |
| No soil moisture | Remove SMAP/ESA CCI | <5% relative drop in primary metrics |
| No irrigation maps | Remove irrigation modality | <3% relative drop (I_latent should compensate) |
| No SAR | Test optical-only fallback | <10% relative drop |
| Cloud-degraded optical | Simulate missing optical observations (mask random time steps) | Graceful degradation, not catastrophic failure |
| Weather-only | No satellite imagery | Documented baseline for zero-optical scenarios |

---

## 12. Baseline Specifications

All baselines must be evaluated on the same splits and with the same statistical protocol (spatial block bootstrap, year-level LOO).

| # | Baseline | Type | Description |
|---|----------|------|-------------|
| 1 | Climatology / Historical Mean | Minimum skill | Average conditions over training period |
| 2 | Persistence Model | Temporal forecasting | Predict current state continues |
| 3 | GDD Phenology Model | Simple crop-physics | Accumulate GDD; predict phenology from thermal thresholds |
| 4 | SPI/SPEI/VHI Drought Index Models | Drought-monitoring | Standard drought indices from climate data |
| 5 | Random Forest / XGBoost | Strong tabular | Hand-crafted features: NDVI trajectory, weather stats, soil properties, crop calendar |
| 6 | LSTM / TCN | Deep temporal | Temporal deep learning on time-series features |
| 7 | Single-modality (optical only) | Ablation | Full model with satellite-only input |
| 8 | Single-modality (weather only) | Ablation | Full model with weather-only input |
| 9 | Single-task (drought only) | Ablation | Full model trained on drought objective only |
| 10 | Single-task (phenology only) | Ablation | Full model trained on phenology objective only |
| 11 | Generic MAE SSL | Ablation | Full model with generic MAE pre-training instead of process SSL |
| 12 | Unconstrained (no physics) | Ablation | Full model without water-balance/growth constraints |
| 13 | Non-causal (no DAG) | Ablation | Full model without temporal DAG constraints |
| 14 | DSSAT/APSIM (optional) | Process-model comparison | Only if detailed management data available |

---

## 13. Code Style Guidelines

- **Standard:** Follow PEP 8 guidelines.
- **Type Hints:** Required for all function and class signatures.
- **Docstrings:** NumPy style docstrings detailing inputs, outputs, and mathematical formulas.
- **Example Snippet (Cross-Attention Fusion):**

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class ScaleAwareFusion(nn.Module):
    """Scale-Aware Cross-Attention Fusion layer.

    Allows fine-resolution satellite tokens to query coarse-resolution
    weather and soil tokens without forcing upsampling.

    Parameters
    ----------
    d_model : int
        Dimension of the model (must be same for fine and coarse tokens).
    nhead : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability for attention weights, by default 0.1.

    Attributes
    ----------
    cross_attn : nn.MultiheadAttention
        Multi-head attention module for cross-attention.
    norm : nn.LayerNorm
        Layer normalization for residual connection.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fine_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        fine_tokens : torch.Tensor
            Shape [B, S_fine, D] — fine-resolution satellite tokens.
        coarse_tokens : torch.Tensor
            Shape [B, S_coarse, D] — coarse-resolution weather/soil tokens.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Fused representation and attention weights.
        """
        attn_output, attn_weights = self.cross_attn(
            query=fine_tokens, key=coarse_tokens, value=coarse_tokens
        )
        return self.norm(fine_tokens + attn_output), attn_weights
```

---

## 14. Testing Strategy

- **Framework:** PyTest.
- **Coverage Target:** Minimum 85% overall coverage, 100% on physics loss formulas.

### 14.1 Unit Test Boundaries

| Test File | What It Tests | Key Assertions |
|-----------|---------------|----------------|
| `test_datacube.py` | Patch loading, missing modalities, data leakage | Data loader masks are strictly zero-leak (no future time steps visible in past inputs); modality masks correctly applied; 64x64 patches have correct coordinate alignment |
| `test_encoders.py` | Encoder dimensions, tensor flow | Forward pass produces correct output shapes; gradient flows through all encoder paths |
| `test_physics.py` | Water-balance constraint, AGDD formula | Water-balance loss is 0.0 when parameters are perfectly balanced; AGDD accumulation matches hand-computed values; Mitscherlich-Baule produces zero growth at zero thermal time or zero moisture |
| `test_diagnostics.py` | Causal DAG, bootstrap | Temporal DAG has no cycles (autocyclicity check); gradient paths do not flow backward in time; spatial block bootstrap produces valid confidence intervals |
| `test_ssl.py` | Collapse monitor, masking | Eigenvalue collapse detection triggers on degenerate representations; masked reconstruction loss ignores unmasked patches; cross-modal prediction uses correct source/target modalities |

### 14.2 Verification Command Mapping

| Sprint | Verification Command | Maps to Test |
|--------|---------------------|--------------|
| 1.1 | `pytest tests/test_datacube.py::test_climate_data_loading` | Validates ERA5/CHIRPS ingestion and 5-day compositing |
| 1.2 | `pytest tests/test_datacube.py::test_satellite_alignment` | Validates S2/Landsat spatial alignment with crop boundaries |
| 1.3 | `pytest tests/test_datacube.py::test_patch_extraction` | Validates 64x64 patch count and metadata |
| 1.4 | `pytest tests/test_datacube.py::test_zero_leakage` | Validates zero data leakage between folds and held-out test sets |
| 2.1 | `pytest tests/test_encoders.py` | Validates all encoder forward passes |
| 2.3 | `pytest tests/test_diagnostics.py::test_causal_dag_acyclicity` | Validates temporal DAG acyclicity |
| 2.4 | `pytest tests/test_physics.py::test_water_balance_zero` | Validates water-balance loss = 0 under perfect conservation |

---

## 15. Boundaries

- **Always do:** Run tests and linter before commits; document every mathematical formula in code; follow PEP 8; use type hints; write NumPy docstrings.
- **Ask first:** Modifying the Spatial-Temporal split years; adding new external data sources; changing physics loss terms; modifying the crop-region matrix; adjusting extreme drought holdout years.
- **Never do:** Resample coarse climate variables to 10m without scale-aware cross-attention; assume irrigation is fully observed without using I_latent balance; use random pixel-level t-tests for EO statistical tests; claim food-security risk from yield-anomaly predictions alone; treat phenology pseudo-labels as ground truth without noise estimation.

---

## 16. Success Criteria (COMPLETE — Including C3 and C6)

| Criterion | Metric | Threshold | Claim Tested |
|-----------|--------|-----------|--------------|
| Model feasibility | Multi-resolution dataloader parses Sentinel-2 (10m) and ERA5 (31km) correctly | Correct token shapes and resolution metadata | — |
| Physics success | Water-budget RMSE reduction vs unconstrained baseline | >= 15% reduction | C4 |
| Physics non-harm | Downstream metric decrease due to physics | <= 3% decrease | C4 |
| Crop Stress AUROC (in-region) | AUROC | >= 0.90 | C1, C2 |
| Crop Stress AUROC (transfer) | AUROC | >= 0.80 | C2, C6 |
| Phenology MAE | Mean absolute error in transition timing | <= 5.0 days | C1 |
| Yield Risk R-squared | District yield anomaly forecast | >= 0.70 on in-region tests | C1, C2 |
| Causal plausibility | Temporal DAG directional sign consistency | 100% consistent with agronomic priors | C5 |
| **Process SSL advantage** | **Linear-probe transfer accuracy / fine-tuning efficiency** | **>= 5% relative improvement over generic MAE on at least 1 primary task or transfer split** | **C3** |
| **Extreme-event robustness** | **Event-level F1 score on drought holdouts** | **>= 10% relative improvement over LSTM/XGBoost baselines** | **C6** |
| Joint modeling benefit | Joint model vs single-task (drought-only, phenology-only) | Statistically meaningful improvement (paired t-test, p < 0.05) on at least 2 of 3 primary tasks | C1 |
| I_latent interpretability | Correlation between I_latent and known irrigation areas | Statistically significant (t-test, p < 0.05) | C4 |
| Drought Trajectory | RMSE vs SPI/SPEI persistence baseline | >= 10% relative RMSE reduction | — |
| Missing modality robustness | Performance drop when removing single modality | <5% relative drop for non-critical modalities; <10% for any single modality | C2 |

---

## 17. Cloud Contamination Protocol

| Scenario | Strategy |
|----------|----------|
| 5-day composite has <20% cloud | Use optical data directly |
| 5-day composite has 20-80% cloud | Flag as partially cloud-contaminated; use with quality weight |
| 5-day composite has >80% cloud | Fall back to SAR/HLS/monthly composite; flag as gap-filled |
| Persistent cloud (>3 consecutive composites) | Use monthly aggregation; report as cloud-gap event |
| Tropical/monsoon regions with chronic cloud | Use SAR as primary; optical as secondary |

Cloud-gap statistics must be reported by region and season as part of the Data Readiness Report.

---

## 18. Storage and Compute Estimates (Updated for 4 Regions x 26 Years with Sparsity)

### 18.1 Storage Estimates

| Data Category | Per Region (26 years, sparse patches) | 4 Regions Total |
|---------------|---------------------------------------|-----------------|
| Sentinel-2 + Landsat (5-day composites, 1,000-5,000 patches x 64x64) | ~300-500 GB | ~1.2-2.0 TB |
| ERA5 climate (daily, multi-variable, coarse) | ~50 GB | ~200 GB |
| CHIRPS precipitation | ~10 GB | ~40 GB |
| SoilGrids (static) | ~5 GB | ~20 GB |
| SMAP soil moisture | ~20 GB | ~80 GB |
| Crop masks and calendars | ~5 GB | ~20 GB |
| Irrigation maps | ~5 GB | ~20 GB |
| Yield labels and auxiliary | ~1 GB | ~4 GB |
| Processed datacubes (sparse patches) | ~100-200 GB | ~400-800 GB |
| **Total** | **~500-800 GB** | **~2.0-3.2 TB** |

**Note:** The sparsity approach (1,000-5,000 patches) reduces the satellite data from ~6 TB (3 regions, 7 years, wall-to-wall) to ~1.2-2.0 TB (4 regions, 26 years, sparse patches). This is a significant reduction while preserving temporal coverage.

### 18.2 Compute Estimates

| Task | Estimated GPU-Hours | Hardware |
|------|---------------------|----------|
| Datacube preprocessing | 200-500 CPU-hours | Multi-core CPU |
| Baseline training (RF/XGBoost) | 50-100 CPU-hours | CPU |
| Baseline training (LSTM/TCN) | 100-300 GPU-hours | 1x A100 |
| SSL pretraining (process-centered, 5 objectives) | 2,000-5,000 GPU-hours | 4-8x A100 |
| Full model training (all heads, joint) | 2,000-4,000 GPU-hours | 8x A100 |
| Physics regularization experiments | 500-1,000 GPU-hours | 4-8x A100 |
| Ablation experiments (13+ baselines) | 1,500-3,000 GPU-hours | 4-8x A100 |
| Transfer and robustness evaluation | 500-1,000 GPU-hours | 4-8x A100 |
| Bootstrap confidence intervals | 200-500 CPU-hours | CPU |
| **Total estimated** | **7,000-15,000 GPU-hours** | **8x A100 (80GB)** |

**Minimum compute requirement:** Access to at least one 8xA100 node for ~3 months of active training.

### 18.3 Reproducibility Requirements

| Requirement | Specification |
|-------------|---------------|
| Random seeds | Fixed seeds for all experiments; logged in config files |
| Software versions | PyTorch, xarray, rasterio versions frozen in requirements.txt |
| Data download scripts | Version-controlled scripts with product IDs and dates |
| Train/val/test split files | Committed to repository BEFORE any model training |
| Model configuration files | YAML/JSON configs for every experiment |
| Experiment tracking | Weights & Biases for all training runs |
| Preprocessing pipeline | Fit on training data only; saved transformers applied to val/test |

---

## 19. Implementation Plan: 52-Week MVRC Roadmap

### Phase 1: Data Engineering & Spatiotemporal Datacube (Weeks 1-14)

| Sprint | Weeks | Tasks | Acceptance | Verify | Gate |
|--------|-------|-------|------------|--------|------|
| 1.1 | 1-3 | Ingest ERA5 weather and CHIRPS rainfall at native resolutions (2000-2026) | Visualized monthly climate climatology matching external databases | `pytest tests/test_datacube.py::test_climate_data_loading` | — |
| 1.2 | 4-7 | Fetch Landsat 5/7/8 (2000-2016) and Sentinel-2 (2017-2026); fetch irrigation maps; handle Landsat 7 SLC-off gaps | Spatial alignment matching crop-field boundaries | `pytest tests/test_datacube.py::test_satellite_alignment` | G0: Scope Lock |
| 1.3 | 8-11 | Extract 1,000-5,000 representative 64x64 agricultural patches; mask clouds and fill gaps | Total database <= 2 TB | `pytest tests/test_datacube.py::test_patch_extraction` | — |
| 1.4 | 12-14 | Implement multi-resolution dataloader; build 5-fold Spatial-Temporal Blocked split indexes | Zero data leakage verified | `pytest tests/test_datacube.py::test_zero_leakage` | G1: Data Readiness |

### Phase 2: Core Architecture & Weak Physics (Weeks 15-28)

| Sprint | Weeks | Tasks | Acceptance | Verify | Gate |
|--------|-------|-------|------------|--------|------|
| 2.1 | 15-18 | Implement spatial ViT, weather transformer, soil MLP, crop calendar embedding, irrigation map encoder with modality masks; integrate AGDD tracking | Successful forward pass with custom tensor sizes | `pytest tests/test_encoders.py` | G2: Baseline Performance |
| 2.2 | 19-21 | Implement cross-attention fusion (fine tokens query coarse context); build hierarchical temporal memory (5-day + monthly pathways) | Successful multi-modal token combination without spatial upsampling | Dimensions of fusion outputs match target hidden sizes | — |
| 2.3 | 22-24 | Design unidirectional temporal DAG pathways; audit gradient flows to prevent retrocausal information leakage | DAG check passes; zero back-in-time gradients | `pytest tests/test_diagnostics.py::test_causal_dag_acyclicity` | G3: Multimodal Fusion Value |
| 2.4 | 25-28 | Code water-balance loss with dynamic lambda scheduling; code AGDD growth constraint | Water-balance loss evaluates to 0.0 under perfect conservation; AGDD matches hand-computed values | `pytest tests/test_physics.py` | — |

### Phase 3: Model Training & Supervised Heads (Weeks 29-42)

| Sprint | Weeks | Tasks | Acceptance | Verify | Gate |
|--------|-------|-------|------------|--------|------|
| 3.1 | 29-33 | Train model on sequential process-centered SSL objectives (SSL-1 through SSL-5); monitor representation collapse | Validation loss converges; embedding variance/rank checks pass | W&B training loss logs | G4: Process SSL Value |
| 3.2 | 34-38 | Attach 3 Primary Heads + 2 Auxiliary Heads; train with joint loss | Model outputs stress probabilities, phenology timing, yield anomalies, drought trajectory, I_latent | Validation metrics evaluated across K-folds | — |
| 3.3 | 39-42 | Run hyperparameter sweeps on learning rates, dropout, physics loss weights (lambda_max), warmup epochs | Optimization achieves high validation stability | W&B sweep logs | G5: Physics Value |

### Phase 4: Rigorous Validation & Claims Audit (Weeks 43-52)

| Sprint | Weeks | Tasks | Acceptance | Verify | Gate |
|--------|-------|-------|------------|--------|------|
| 4.1 | 43-46 | Apply Spatial Block Bootstrap (50km) and Year-Level LOO Bootstrap; calculate 95% CIs and p-values | Valid, uninflated statistical tests for all downstream heads | `python scripts/evaluate_model.py --run_bootstrap` | — |
| 4.2 | 47-49 | Evaluate on held-out extreme drought years (2005 Spain, 2012 US, 2022 IGP); run cross-region transfer (Wheat: IGP <-> Spain; Maize: US <-> Po Valley) | Target metrics achieved (Stress AUROC >= 0.90 in-region, >= 0.80 transfer; Yield R-squared >= 0.70; Event F1 >= 10% improvement) | Generate performance tables | G6: Causal-Informed Value |
| 4.3 | 50-51 | Verify water constraints reduce RMSE >= 15% with <= 3% downstream loss; verify causal direction sign correctness | Physical and causal diagnostics pass | `python scripts/evaluate_model.py --run_physics_causal_audit` | — |
| 4.4 | 52 | Baseline comparison finalization; ablation table assembly; error analysis; manuscript-ready results | Ablation tables complete; all claims C1-C6 documented (positive or negative) | Manual review | G7: Research Readiness |

---

## 20. Implementation Gates

| Gate | Name | Go Condition | No-Go Condition | Decision |
|------|------|-------------|-----------------|----------|
| G0 | Scope Lock | Final crop/region/year/task/metric/data selection is testable and data exists | No available data for any proposed region-crop combination | Redefine scope; select different regions |
| G1 | Data Readiness | At least 2 regions have sufficient satellite/weather/soil/crop/yield data; cloud/missingness profile acceptable | No reliable labels or severe missingness with no fallback | Fix data pipeline; select alternative regions |
| G2 | Baseline Performance | At least one simple baseline beats climatology/persistence; evaluation pipeline stable | Labels or task definition appear invalid | Revisit task formulation; check labels |
| G3 | Multimodal Fusion Value | Fusion improves over best single-modality baseline in at least 1 primary task or transfer split | Fusion adds no value after alignment/missingness fixes | Analyze attention patterns; try alternative fusion |
| G4 | Process SSL Value | Process SSL improves transfer, fine-tuning efficiency, or extreme-event performance over generic MAE | SSL collapses or provides no measurable benefit | Revert to generic SSL; report as negative finding |
| G5 | Physics Value | Physics improves consistency (>=15% residual reduction) without major task degradation (<=3% drop) | Physics worsens both residuals and task metrics | Report physics as diagnostic only |
| G6 | Causal-Informed Value | Temporal-DAG passes sign, monotonicity, and invariance checks | Scenario behavior is unstable or contradicts domain knowledge | Treat causal module as interpretability only |
| G7 | Research Readiness | Ablations support at least 1 clear contribution; results reproducible; failure modes documented | Results are broad but inconclusive | Extend analysis; target specific contribution |

---

## 21. Skills Mapping (Phase & Task Wise)

| Skill / Subagent | Description | Phase | Tasks / Sprints |
|------------------|-------------|-------|-----------------|
| `data-quality-auditor` | Profile datasets, verify cloud/gap masks, audit data splits | Phase 1 & 3 | Sprints 1.1-1.3 (Verify satellite and weather datasets), Sprint 3.1 (Verify missing-modality dropout) |
| `senior-ml-engineer` | Ingest multi-resolution data, build PyTorch datasets/dataloaders, construct model encoders and fusion layers | Phase 1 & 2 | Sprints 1.4 (dataloader), 2.1 (encoders), 2.2 (fusion) |
| `senior-computer-vision` | Design spatial ViT, patch-processing, masked autoencoders (SSL) | Phase 2 & 3 | Sprints 2.1 (spatial ViT), 3.1 (SSL pre-training) |
| `senior-data-scientist` | Implement physics regularization, temporal DAGs, spatial block bootstrap, and statistical hypothesis tests | Phase 2 & 4 | Sprints 2.3 (temporal DAG), 2.4 (physics), 4.1 (bootstrap), 4.3 (physics auditing) |
| `autoresearch-agent` | Perform hyperparameter tuning sweeps, evaluate baselines, and format paper reports | Phase 3 & 4 | Sprints 3.2-3.3 (task training, sweeps), 4.2 (extreme-event evaluation), 4.4 (baseline comparison & finalization) |

---

## 22. Risk Register

| ID | Risk | Severity | Impact | Mitigation | Fallback |
|----|------|----------|--------|------------|----------|
| R1 | Overbroad foundation-model scope | High | Could prevent completion | MVRC defined for 52 weeks | Narrow scope further; defer modules |
| R2 | Data availability gaps (especially pre-Sentinel era) | High | Reduced temporal coverage or missing modalities | Document data availability by era; use modality masks | Restrict to 2017+ if pre-Sentinel data proves unusable |
| R3 | Representation collapse during SSL | High | SSL fails to learn useful representations | Collapse monitoring (eigenvalue threshold); temperature scheduling | Revert to generic MAE; report process SSL as negative finding |
| R4 | Physics loss destabilizes training | Medium | Task performance degrades | Dynamic lambda scheduling with warmup | Remove physics; report as diagnostic finding |
| R5 | Missing irrigation data confounds relationships | High | I_latent uninterpretable | I_latent validated against known irrigation areas | Treat irrigation as documented confounder |
| R6 | Phenology pseudo-label noise too high | Medium | Phenology head unreliable | Noise estimation; soft labels; multiple index agreement | Report phenology as exploratory; focus on stress and yield |
| R7 | Compute budget exceeded | Medium | Cannot complete all experiments | Prioritize C1-C3 ablations; reduce patch count | Use subset of patches (1,000 instead of 5,000) |
| R8 | I_latent is uninterpretable | Medium | Cannot validate auxiliary head | Compare with irrigation maps and canal command areas | Remove I_latent; treat irrigation as confounder |
| R9 | Cross-region transfer fails | Medium | C2 and C6 claims unsupported | Check alignment, missing data, attention collapse | Report negative transfer finding |
| R10 | Causal DAG passes sign checks but fails invariance | Low | C5 cannot be claimed as performance contribution | Treat as interpretability contribution only | Report as analysis, not as model feature |

---

## 23. Terminology Compliance

| Allowed Terminology | Prohibited Terminology (in MVRC) | Reason |
|--------------------|--------------------------------|--------|
| Crop-climate risk | Food security risk | Requires socioeconomic data not included |
| Yield anomaly / yield shortfall | Food shortage | Implies broader socioeconomic impact |
| Crop stress detection | Famine prediction | Not supported by model scope |
| Agricultural production risk | Food crisis | Overreaches model capabilities |
| I_latent (latent residual water input) | Irrigation detection | I_latent captures multiple unobserved water sources, not just irrigation |
| Process-centered SSL | General self-supervised learning | Specify the process-centered nature |
| Weak physics regularization | Physics-informed model | "Weak" acknowledges incompleteness |
| Temporal DAG constraint | Causal discovery | Model uses structured constraints, not full discovery |

---

## 24. Documentation Cleanup & Hygiene (Updated for v2.2)

The following documents from previous iterations are superseded by this v2.2 specification:

| Previous Document | Status | Action |
|-------------------|--------|--------|
| `AgroEarthFM_Complete_Finalized_Specification.md` (v2.0) | Superseded | Archive; this v2.2 document is the authoritative source |
| `AgroEarthFM_Next_Steps_Action_Plan.docx` | Superseded | Archive; all resolutions incorporated into v2.2 |
| `AgroEarthFM_Scope_Lock_Data_Readiness_Spec.docx` | Superseded | Archive; all scope-lock artifacts incorporated into v2.2 |
| `AgroEarthFM_Implementation_Plan.docx` (v1.0) | Superseded | Archive; original plan replaced by v2.2 roadmap |

---

## 25. Validation Summary

This v2.2 specification has been validated against all prior documents and external reviews:

1. **Original Implementation Plan** (v1.0): Core architecture, multimodal inputs, and research direction are carried forward.
2. **External Validation Round 1** (6.5/10): All 4 blockers resolved (DAG cycle -> temporal DAG; timeline realism -> 52-week MVRC; irrigation gap -> I_latent; food security overreach -> crop-climate risk).
3. **Next Steps Action Plan**: All resolutions incorporated (revised hypothesis, architecture, validation matrix, 52-week roadmap, 21 action items).
4. **External Validation Round 2**: Remaining gaps (exact specifications) addressed with concrete tables and thresholds throughout this document.
5. **v2.1 Review (this session)**: 4 critical issues, 6 high-priority issues, 8 medium-priority issues, and 7 low-priority issues identified and resolved in this v2.2 document.

**Key corrections from v2.1 to v2.2:**
- Irrigation policy reverted from "required input" to "available covariate with modality masks" (preserving I_latent design)
- 2022 IGP data split conflict resolved (extreme holdout years excluded from normal test sets)
- Data Availability by Era table added for 2000-2026 expansion
- Success criteria completed for C3 (Process SSL) and C6 (Tail Events)
- Sprint 4.4 added to timeline for baseline comparison and finalization
- Po Valley data availability verified with explicit table
- Missing project structure modules added (ssl/, baselines/, modality_masks.py, dynamic_scheduler.py)
- I_latent role clarified as "residual water input NOT captured by available irrigation maps"
- Storage/compute estimates updated for 4 regions x 26 years with sparsity
- Cross-platform activation commands added
- Extreme drought year for Spain corrected from 2003 to 2005
- Baseline specifications enumerated (14 baselines)
- Dynamic lambda scheduling specified
- Verification commands mapped to actual test functions
- Label noise estimation required
- AGDD and Mitscherlich-Baule formulas specified
- Implementation gates (G0-G7) integrated into phase structure
- Cloud contamination protocol included
- Missing-modality test matrix included
- Scale compliance rule included
- No-leakage protocol included
- Risk register included
- Terminology compliance table included
