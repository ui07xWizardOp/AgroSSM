# AgroEarthFM: Complete Finalized Specification

## Scope-Lock, Data-Readiness, and Comprehensive Research Blueprint

**Document Version:** 2.0 (Finalized)  
**Date:** 2026-05-22  
**Status:** Pre-Implementation Specification  
**Classification:** Research Prototype Specification  

---

# PART I: RESEARCH FOUNDATION

---

## 1. Research Vision and Background

### 1.1 The Scientific Gap

Current geospatial foundation models (GeoFMs) such as SatMAE, DOFA, PhilEO, and TerraFM learn powerful visual representations from Earth observation data, but they are fundamentally limited in the agro-ecosystem domain for three critical reasons. First, they learn appearance-based representations optimized for image reconstruction or contrastive objectives, not process-centered representations that understand how drought stress propagates through soil moisture decline, vegetation stress, phenology disruption, and ultimately crop yield reduction. Second, they treat each pixel or patch independently, ignoring the temporal causal chain that connects climate forcing to agricultural outcomes. Third, they lack physics-awareness — they can detect that a field looks stressed but cannot enforce or verify that their representations are consistent with basic water-balance or plant-growth constraints.

The central innovation of AgroEarthFM is to move beyond appearance-based geospatial representation learning toward **process-centered representation learning** that jointly models drought dynamics and crop phenology within physical and causal constraints. This means the model does not merely learn to reconstruct satellite imagery — it learns to predict how drought stress evolves over time, how phenology transitions respond to environmental forcing, and how these processes interact under the governing constraints of water balance and plant physiology.

### 1.2 Research Direction

AgroEarthFM merges two research directions that have traditionally been treated separately:

**Direction A — Drought Modeling:** Remote sensing and climate data are used to detect, monitor, and forecast drought conditions, typically through indices such as SPI, SPEI, VHI, and soil moisture anomalies. These approaches focus on the climate-hydrology-vegetation chain but rarely model crop-specific phenology or yield implications.

**Direction B — Crop Phenology Modeling:** Time-series of vegetation indices are used to detect phenological transitions (green-up, heading, maturity, senescence) and predict crop development. These approaches focus on crop-specific temporal patterns but rarely incorporate drought dynamics as a process driver.

AgroEarthFM unifies these directions by learning a shared multimodal temporal representation that captures both drought stress evolution and crop phenology dynamics as interacting processes, constrained by physical laws and structured by causal relationships.

### 1.3 Long-Term Vision (Phase 2 and Beyond)

The long-term vision (24-month horizon) extends AgroEarthFM to include: full neural causal discovery with validated counterfactual reasoning, food-security risk assessment requiring socioeconomic vulnerability and exposure data, global multi-biome generalization across diverse agro-climatic zones, integration with process-based crop simulation models (DSSAT/APSIM/AquaCrop) for hybrid physics-data modeling, and operational decision-support deployment for agricultural resilience planning. These are explicitly **deferred** from the first-year MVRC and will be pursued only after the core scientific contribution is validated.

---

## 2. Revised Core Hypothesis

### 2.1 Final Revised Hypothesis

> A multimodal temporal representation trained on joint drought and crop-phenology process objectives, and regularized by weak water-balance constraints, will improve crop-stress detection, phenology forecasting, and yield-risk prediction under out-of-region, out-of-year, and extreme-event validation compared with single-domain, non-physics, and conventional machine-learning baselines.

### 2.2 Hypothesis Rationale

This hypothesis is structured as a conditional causal claim: IF the model is trained with joint process objectives AND regularized by weak physics constraints, THEN it will outperform baselines on specific evaluation settings. The "weak" qualifier on physics constraints is essential — the water-balance constraint is incomplete (missing irrigation, groundwater, management), and the plant-growth constraint is approximate (Mitscherlich-Baule is a simplified growth response). The hypothesis claims that even incomplete physics, applied as soft regularization with uncertainty weighting, provides measurable benefit over unconstrained models.

### 2.3 Hypothesis Scope Boundary

The hypothesis does NOT claim:
- Full causal discovery from observational data alone
- Food-security risk prediction (requires socioeconomic data not included)
- Global generalization across all climate zones
- Decision-grade counterfactual estimates
- Field-level yield accuracy from administrative-level yield labels

---

## 3. Atomic Testable Claims

The hypothesis is decomposed into six atomic testable claims, each mapped to a specific validation experiment, required baseline, success signal, and failure action.

### 3.1 Claim Register

| Claim ID | Claim | Validation Requirement | Required Baseline | Success Signal | Failure Action |
|---|---|---|---|---|---|
| C1 | Joint drought + phenology modeling improves representation quality | Train joint model vs drought-only and phenology-only models | Single-domain SSL models | Statistically meaningful improvement on stress, phenology, and yield tasks (paired t-test, p < 0.05) | Reduce model complexity; inspect whether tasks conflict |
| C2 | Multimodal fusion improves performance and robustness | Compare satellite-only, weather-only, soil-only, and fused models | Best single-modality model | Fused model improves OOD and extreme-event performance by at least 5% relative improvement | Check alignment, missing data, attention collapse |
| C3 | Process-centered SSL improves transfer over generic SSL | Compare process SSL against generic MAE/contrastive SSL and supervised-from-scratch baselines | Generic MAE + supervised baseline | Improved linear-probe transfer, fine-tuning efficiency, or extreme-event recall by measurable margin | Revert to generic SSL; report process SSL as negative finding |
| C4 | Weak physics regularization improves physical consistency without harming task skill | Compare with and without water-balance/growth constraints | Same model without physics | Lower physical residuals (>=15% reduction in water-balance RMSE) without major task degradation (<=3% drop in primary metrics) | Reduce physics weight; add residual/unobserved flux term; report as diagnostic |
| C5 | Temporal causally informed structure improves interpretability and scenario behavior | Validate through temporal-DAG sign checks, invariance tests, and known-event sensitivity tests | Multimodal non-causal model | Correct directional signs, stable invariance across regions, plausible scenario responses | Treat causal layer as interpretability only, not performance claim |
| C6 | The model performs better on difficult conditions | Evaluate on out-of-region, out-of-year, and drought-event holdouts | LSTM/XGBoost/task-specific baselines | Improved recall/skill on rare severe events (>=10% relative improvement in event-level F1) | Rebalance training, add event-focused objectives |

### 3.2 Claim Validation Priority

| Priority | Claims | Justification |
|---|---|---|
| Must-validate | C1, C2, C3 | Core novelty of the paper |
| Strongly-validate | C4, C6 | Key differentiators |
| Validate-if-possible | C5 | High-risk, high-reward; acceptable as interpretability contribution |

---

## 4. Causal Chain and Temporal DAG

### 4.1 The Agro-Ecosystem Process Chain

The fundamental process chain that AgroEarthFM models is:

```
Climate Forcing → Temperature Anomalies → ET Imbalance → Soil Moisture Decline
→ Vegetation Stress → Phenology Disruption → Crop Stress Accumulation → Yield Reduction
```

This chain represents the physical pathway through which drought impacts propagate through the agro-ecosystem. Each link in this chain is grounded in established agricultural and hydrological science.

### 4.2 Original Causal Graph (DEFECTIVE — Contains DAG Cycle)

The original implementation plan proposed a static causal graph with a bidirectional edge between Vegetation Health (VH) and Crop Stage (CS):

```
VH ↔ CS  (bidirectional, same time step)
```

This violates the Directed Acyclic Graph (DAG) requirement because it creates a simultaneous cycle: VH influences CS and CS influences VH at the same time step, making causal identification impossible.

**Blocker Status:** RESOLVED — replaced with temporal DAG.

### 4.3 Revised Temporal DAG (FINAL)

The revised causal structure is time-indexed, eliminating same-time-step cycles:

```
Weather(t) → ET(t)
Rainfall(t), ET(t), Soil(t), I_latent(t) → Soil_Moisture(t)
Soil_Moisture(t), Temperature(t), Crop_Stage(t) → Vegetation_Health(t)
Vegetation_Health(t), AGDD(t), Crop_Calendar → Crop_Stage(t+1)
Vegetation_Health(t), Crop_Stage(t), Stress_Accumulation(t) → Yield_Risk(t+1)
```

### 4.4 Temporal DAG Visual Specification

```
Rainfall_t ───────────────┐
                          ▼
Temperature_t ──────► ET_t ─────► Soil_Moisture_t
      │                                │
      │                                ▼
      └──────────────► Vegetation_Health_t
                                       │
Crop_Stage_t ─────────────────────────┘
      │                                │
      ▼                                ▼
Crop_Stage_t+1 ◄──────── Vegetation_Health_t
      │
      ▼
Yield_Risk_t+1
```

### 4.5 DAG Structural Properties

| Property | Specification |
|---|---|
| Acyclicity | No same-time directed cycles; temporal indexing ensures DAG validity |
| Edge directionality | All edges are temporally forward or same-time with clear causal precedence |
| Vegetation Health → Crop Stage | Forward in time: VH(t) → CS(t+1) |
| Crop Stage → Vegetation Health | Same time step: CS(t) → VH(t) (crop stage determines stress sensitivity) |
| I_latent role | Latent unobserved water input enters soil moisture equation at time t |
| Stress accumulation | Cumulative term that accumulates across time steps |

### 4.6 Key Causal Relationships Explained

**Weather → ET**: Temperature and radiation drive evapotranspiration through the Penman-Monteith relationship. Higher temperatures increase atmospheric demand for water vapor, accelerating ET rates.

**Rainfall + ET + I_latent → Soil Moisture**: The water balance determines soil moisture state. Rainfall and latent water inputs (irrigation, groundwater) add water; ET and runoff remove it. The I_latent term captures all unobserved water inputs.

**Soil Moisture + Temperature + Crop Stage → Vegetation Health**: Vegetation health at time t depends on water availability (soil moisture), atmospheric stress (temperature), and the crop's current development stage, which determines its stress sensitivity. Crops are most sensitive to drought during flowering and grain-filling stages.

**Vegetation Health + AGDD + Crop Calendar → Crop Stage (t+1)**: Phenology advances based on accumulated thermal time (Growing Degree Days) and the crop's current health status. Stressed crops may experience delayed phenology. The crop calendar provides a prior for expected transition timing.

**Vegetation Health + Crop Stage + Stress Accumulation → Yield Risk (t+1)**: Yield risk emerges from the cumulative stress experienced by the crop, weighted by the phenology stage at which stress occurred. Stress during reproductive stages has disproportionate yield impact.

---

# PART II: ARCHITECTURE SPECIFICATION

---

## 5. Complete Model Architecture

### 5.1 MVP Architecture (Final)

The first implementation uses a staged architecture with the following components:

```
Fine Satellite Encoder (ViT-based)
        │
Weather / Climate Temporal Encoder (Temporal Transformer)
        │
Soil / Static Context Encoder (MLP / GNN)
        │
Crop Calendar / Crop Type Embedding (Learned Embedding)
        │
Scale-Aware Fusion Module (Cross-Attention with Resolution Awareness)
        │
Hierarchical Temporal Memory Module (5-day + Monthly)
        │
Shared Agro-Ecosystem State Representation
        │
 ┌───────────────┬────────────────┬────────────────┐
 │ Crop Stress   │ Phenology       │ Yield Risk /   │
 │ Head          │ Forecast Head   │ Yield Anomaly  │
 └───────────────┴────────────────┴────────────────┘
        │
Weak Physics Regularization (Water Balance + Plant Growth) during training
```

### 5.2 Component Specifications

#### 5.2.1 Fine Satellite Encoder

| Parameter | Specification |
|---|---|
| Base architecture | Vision Transformer (ViT) |
| Input | Sentinel-2 / HLS multi-spectral imagery (10-30m) |
| Patch size | 16×16 pixels |
| Input bands | B2, B3, B4, B8 (10m) + B5, B6, B7, B8A, B11, B12 (20m resampled) |
| Temporal input | Sequence of 5-day composites |
| Positional encoding | 2D spatial + temporal encoding |
| Output | Token sequence per time step |
| Pretraining | SSL objectives (masked spatiotemporal reconstruction) |
| Approximate parameter count | 30-50M parameters |

**Design rationale:** The ViT architecture is chosen because it produces spatially explicit token representations that can be attended to by subsequent fusion and temporal modules. Patch size of 16×16 at 10m resolution corresponds to 160m ground patches, providing a balance between spatial detail and computational tractability.

#### 5.2.2 Weather / Climate Temporal Encoder

| Parameter | Specification |
|---|---|
| Base architecture | Temporal Transformer |
| Input | ERA5 / CHIRPS climate variables at native resolution (~31km / ~5km) |
| Variables | Temperature (min/max/mean), precipitation, wind speed, solar radiation, humidity |
| Temporal resolution | Daily aggregated to 5-day windows |
| Positional encoding | Temporal encoding with day-of-year and season indicators |
| Output | Climate state vector per time step |
| Approximate parameter count | 10-15M parameters |

**Design rationale:** Climate variables operate at much coarser spatial resolution than satellite data but carry essential forcing information. The temporal transformer captures weather sequences and their evolution, which is critical for drought onset and development modeling.

#### 5.2.3 Soil / Static Context Encoder

| Parameter | Specification |
|---|---|
| Base architecture | MLP or single Graph Neural Network (GNN) if topology available |
| Input | SoilGrids variables at ~250m resolution |
| Variables | Clay content, sand content, organic carbon, pH, CEC, bulk density, soil depth |
| Spatial context | If GNN used: adjacency from DEM/drainage/field boundaries |
| Output | Static soil-context vector |
| Approximate parameter count | 2-5M parameters |

**Resolution of redundancy:** The original plan had both a soil GNN and a spatial-topology GNN. These are unified into a single spatial-context encoder where soil properties become node features and topology/adjacency/slope/hydrology become edge features. If graph construction is unreliable, use patch-level static soil/topography embeddings instead.

#### 5.2.4 Crop Calendar / Crop Type Embedding

| Parameter | Specification |
|---|---|
| Architecture | Learned embedding layer |
| Input | Crop type (one-hot or embedding index), planting date, expected harvest date |
| Source | GEOGLAM crop calendars, crop masks |
| Output | Crop-context vector per pixel/patch |
| Approximate parameter count | <1M parameters |

**Design rationale:** Crop calendar provides essential prior information about expected phenology timing. The embedding allows the model to learn crop-specific stress responses and phenology patterns without hard-coding rules.

#### 5.2.5 Scale-Aware Fusion Module

| Parameter | Specification |
|---|---|
| Architecture | Cross-attention with resolution-aware positional encoding |
| Input | Token sequences from all encoders |
| Mechanism | Fine-resolution tokens attend to coarse-resolution context |
| Resolution handling | Multi-resolution tokens preserved; no forced upsampling to 10m |
| Missing modality | Modality-present mask + learned missing-modality token |
| Output | Fused multimodal representation per spatial-temporal position |
| Approximate parameter count | 15-25M parameters |

**Critical design decision:** This module implements the multi-resolution fusion principle. Coarse-resolution variables (ERA5, SMAP, GRACE) are NOT upsampled to 10m. Instead, the cross-attention mechanism allows fine-resolution tokens to query coarse-resolution context, and coarse-resolution tokens to aggregate fine-resolution detail. This avoids false precision from resampling.

#### 5.2.6 Hierarchical Temporal Memory Module

| Parameter | Specification |
|---|---|
| Architecture | Two-level temporal transformer |
| Level 1 | Short-term memory: 5-day composite sequence (captures within-season dynamics) |
| Level 2 | Long-term memory: Monthly aggregated sequence (captures seasonal patterns) |
| Input | Fused multimodal token sequence |
| Output | Temporal state representation at each time step |
| Approximate parameter count | 20-30M parameters |

**Design rationale:** Drought develops over weeks to months, while phenology transitions occur over days to weeks. The two-level structure captures both timescales without requiring extremely long sequences at fine temporal resolution, which would be computationally prohibitive.

#### 5.2.7 Output Heads

| Head | Architecture | Output |
|---|---|---|
| Crop Stress Detection Head | MLP with sigmoid output | Binary/multi-class stress probability per pixel per time step |
| Phenology Forecast Head | MLP with softmax over phenology stages | Stage probability + transition timing (MAE in days) |
| Yield Risk / Yield Anomaly Head | MLP with Gaussian output | Yield anomaly prediction with uncertainty |

**Note:** The Food-Security Risk head is REMOVED from the MVP. Food-security risk requires socioeconomic vulnerability, market, storage, and population data that are not included in the first model. This has been reframed as "agricultural production risk" or "yield-shortfall risk."

### 5.3 Components Deferred from MVP

| Component | Status | Reason | Re-evaluation Condition |
|---|---|---|---|
| Full spatial topology GNN | Optional extension | Requires robust graph construction from DEM/drainage/field adjacency | If spatial context encoding proves insufficient with MLP alone |
| Full neural causal discovery | Deferred (Phase 2) | High validation burden; learnable edge strengths sufficient for MVP | After causal-informed module passes G6 gate |
| Counterfactual reasoning module | Deferred or limited demo (Phase 2) | Requires validated temporal-DAG and physics layer | After C5 is validated and physics is stable |
| Food-security output head | Removed from MVP | Requires socioeconomic data not included | If socioeconomic data pipeline is built in Phase 2 |
| Resilience score head | Deferred | Requires validated stress + recovery modeling | After crop-stress detection is robust |
| DSSAT/APSIM integration | Optional case-study baseline | Requires detailed management/calibration data | If process-model comparison becomes essential for paper claims |

### 5.4 Model Size and Token Budget

| Component | Estimated Parameters | Token Budget |
|---|---|---|
| Satellite Encoder (ViT) | 30-50M | ~256 tokens per time step (16×16 patches in 256×256 input) |
| Weather Encoder | 10-15M | ~1-4 tokens per time step (coarse resolution) |
| Soil Encoder | 2-5M | ~1-4 tokens per time step (static, 250m) |
| Crop Calendar Embedding | <1M | ~1 token per time step |
| Fusion Module | 15-25M | Cross-attention queries: fine tokens; keys/values: all tokens |
| Temporal Memory | 20-30M | 5-day: ~24 tokens (120 days); Monthly: ~6 tokens (6 months) |
| Output Heads | 2-5M | Per-head MLP |
| **Total MVP** | **~80-130M** | **~280-290 tokens per time step** |

**Compute consideration:** This model size is designed to be trainable on a single 8×A100 (80GB) GPU node. The token budget is constrained to ensure batch sizes of at least 4-8 per GPU during SSL pretraining.

---

## 6. Missing Modality Protocol

### 6.1 Design

The model must support missing or unavailable inputs during both training and inference.

For each modality:
1. Add a **modality-present mask** (binary indicator per sample per modality)
2. Add a **learned missing-modality token** that replaces absent modality tokens
3. Use **modality dropout** during training (randomly zero out entire modalities with probability p=0.1-0.2)
4. Report performance under missing-modality scenarios

### 6.2 Missing-Modality Test Matrix

| Test | Description | Acceptance Criterion |
|---|---|---|
| Full modality | All inputs available | Baseline performance |
| No soil moisture | Remove SMAP/ESA CCI | <5% relative drop in primary metrics |
| No irrigation proxy | Remove irrigation-related features | Documented performance impact |
| No SAR (if SAR included) | Test optical-only fallback | <10% relative drop |
| Cloud-degraded optical | Simulate missing optical observations (mask random time steps) | Graceful degradation, not catastrophic failure |
| Weather-only | No satellite imagery | Documented baseline for zero-optical scenarios |

---

# PART III: DATA STRATEGY

---

## 7. Multi-Resolution Datacube Design

### 7.1 Core Principle

The model does NOT force all data to a common 10m grid. Each variable is represented at its appropriate spatial and temporal scale. The fusion module is designed to handle multi-resolution inputs natively.

The original plan's approach of resampling all data to 10m Sentinel-2 resolution creates visually detailed but physically false precision. ERA5, SMAP, GRACE, groundwater, administrative yield, and management data do not support true 10m inference. Resampling them to 10m creates spurious spatial detail that can mislead both the model and the evaluator.

### 7.2 Resolution Tiers

| Tier | Spatial Scale | Temporal Scale | Data Types | Token Treatment |
|---|---|---|---|---|
| Fine | 10-30m | 5-day composites | Sentinel-2/HLS reflectance, vegetation indices, SAR | Fine-resolution visual/vegetation tokens |
| Intermediate | 100-500m | 5-day to monthly | Aggregated vegetation dynamics, Sentinel/SAR-derived moisture proxies | Aggregated context tokens |
| Coarse | 1-10km | Daily to monthly | ERA5/CHIRPS weather, SMAP/ESA CCI soil moisture, drought indices | Coarse temporal weather/moisture tokens |
| Regional | Admin/field level | Seasonal/annual | Yield statistics, management records, crop maps | Regional context tokens |

### 7.3 Datacube Construction Protocol

1. **Coregistration:** All data within each tier is reprojected to a common CRS (EPSG:4326 or UTM zone-specific)
2. **Temporal alignment:** All data is aligned to 5-day composite windows (Day 1-5, 6-10, ..., 361-365)
3. **Cloud handling:** 5-day composites use median compositing with valid-pixel filtering; cloud threshold documented per region/season
4. **No cross-tier resampling:** Fine-tier data is NOT upsampled from coarse-tier; coarse-tier data is NOT forced to 10m
5. **Metadata preservation:** Each variable retains its native resolution, uncertainty estimate, and source metadata

---

## 8. Complete Data Source Catalog

### 8.1 Satellite Imagery

| Source | Product | Resolution | Bands | Access | License |
|---|---|---|---|---|---|
| Sentinel-2 | L2A reflectance | 10-20m | B2-B12, B8A | Google Earth Engine / Copernicus Open Access Hub | CC-BY-SA 4.0 / Free |
| HLS (Harmonized Landsat Sentinel-2) | HLSL30/HLSS30 | 30m | Visual + NIR + SWIR | LP DAAC / Earth Engine | Public domain |
| Landsat 8/9 | L2 reflectance | 30m | Multi-spectral | USGS Earth Explorer / Earth Engine | Public domain |

**Sentinel-2 bands used:**
- 10m: B2 (490nm Blue), B3 (560nm Green), B4 (665nm Red), B8 (842nm NIR)
- 20m (resampled to 10m for patch-based processing): B5 (705nm Red Edge 1), B6 (740nm Red Edge 2), B7 (783nm Red Edge 3), B8A (865nm Narrow NIR), B11 (1610nm SWIR 1), B12 (2190nm SWIR 2)

### 8.2 Vegetation Indices

| Index | Formula | Purpose | Derivation |
|---|---|---|---|
| NDVI | (B8 - B4) / (B8 + B4) | Green vegetation density | From S2 reflectance |
| EVI | 2.5 × (B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 1) | Corrected vegetation density | From S2 reflectance |
| NDWI | (B8 - B11) / (B8 + B11) | Vegetation water content | From S2 reflectance |
| LAI | From PROSAIL inversion or empirical | Leaf area index | From S2 reflectance |
| FAPAR | From PROSAIL inversion or empirical | Absorbed photosynthetically active radiation | From S2 reflectance |
| NDMI | (B8 - B11) / (B8 + B11) | Normalized difference moisture index | From S2 reflectance |

**Note on pseudo-labels:** Phenology transitions derived from vegetation-index derivatives are noisy pseudo-labels. The plan explicitly acknowledges this and estimates label noise rather than treating them as ground truth.

### 8.3 Climate Data

| Source | Variables | Spatial Resolution | Temporal Resolution | Access | License |
|---|---|---|---|---|---|
| ERA5 (ECMWF) | T_min, T_max, T_mean, P, wind, RH, solar radiation, ET0 | ~31km | Hourly → daily → 5-day | CDS API / Earth Engine | CC-BY-4.0 |
| CHIRPS | Precipitation | ~5km | Daily | CHIRPS API / Earth Engine | Public domain |
| ERA5-Land | Enhanced land variables | ~9km | Hourly | CDS API | CC-BY-4.0 |

**ERA5 variables extracted:**
- 2m temperature (min, max, mean over 5-day window)
- Total precipitation (sum over 5-day window)
- 10m wind speed (u and v components)
- Relative humidity
- Surface solar radiation downwards
- Potential evapotranspiration (ET0 from FAO Penman-Monteith)

### 8.4 Soil Data

| Source | Variables | Resolution | Type | Access |
|---|---|---|---|---|
| SoilGrids 250m | Clay %, Sand %, Organic C, pH, CEC, Bulk density, Depth | ~250m | Static | ISRIC / Earth Engine |
| ISRIC World Soil Information | Soil taxonomy, water holding capacity | Variable | Static | ISRIC |

### 8.5 Soil Moisture

| Source | Product | Resolution | Depth | Access |
|---|---|---|---|---|
| SMAP | L3 Enhanced | ~9km (enhanced to ~3km) | 0-5cm | NASA DAAC / Earth Engine |
| ESA CCI Soil Moisture | Active + Passive merged | ~25km | 0-5cm | ESA CCI / Earth Engine |

**Critical note:** Soil moisture products provide coarse-resolution surface (0-5cm) estimates. They represent large-area moisture conditions, not field-scale root-zone soil moisture. They should be treated as coarse context, not fine-grained truth.

### 8.6 Groundwater

| Source | Product | Resolution | Access |
|---|---|---|---|
| GRACE/GRACE-FO | Liquid water equivalent thickness | ~150-300km | NASA JPL / Earth Engine |

**Status:** Optional regional context only. Resolution is far too coarse for field-scale modeling. Used only as a regional drought indicator if included at all.

### 8.7 Crop Calendar and Crop Mask

| Source | Variables | Resolution | Access |
|---|---|---|---|
| GEOGLAM Crop Calendar | Planting/harvest dates by crop and region | Regional/country | GEOGLAM / USDA |
| ESA WorldCereal | Crop type maps | 10m | ESA |
| CDL (US, if US region used) | Crop type | 30m | USDA NASS |
| Monsoon Crop Maps (India, if IGP region used) | Crop type, kharif/rabi | State/district | Government of India / ISRO |

**Crop mask errors:** Crop-type maps contain errors and should be treated as noisy labels. Use confidence filtering and crop-mask uncertainty in the pipeline. Sensitivity testing with high-confidence crop pixels only is required.

### 8.8 Yield Labels

| Source | Scale | Access | Quality |
|---|---|---|---|
| USDA NASS (US) | County-level | Public | High quality, annually updated |
| ICRISAT (India) | District-level | Public | Good quality, historical series |
| FAOSTAT | Country-level | Public | Coarse, useful for cross-country comparison |
| Field-level surveys | Field-level | Varies | Rare but valuable when available |

**Scale mismatch rule:** If yield is available only at district/county level, field-level predictions must be aggregated to the label scale before validation. Never claim field-level yield accuracy from administrative-level labels.

### 8.9 Irrigation and Management Proxies

| Source | Data Type | Resolution | Access |
|---|---|---|---|
| GMIA (FAO) | Irrigated area fraction | ~10km | FAO |
| IWMI Global Irrigation Map | Irrigated/rainfed classification | ~500m | IWMI |
| ESA WorldCereal | Irrigation classification | 10m | ESA |
| Census data | Irrigated area by district | Admin level | Government statistical offices |

**Critical decision:** Direct irrigation data is unreliable at fine spatial scales. The model uses I_latent (latent residual water-input inference term) rather than attempting to directly detect irrigation. Irrigation maps are used only as validation references for I_latent, not as direct inputs.

---

## 9. Scope-Lock Specifications

### 9.1 Crop Selection

| Crop | Justification |
|---|---|
| Wheat | Globally important cereal; well-studied phenology; extensive data availability; drought-sensitive during reproductive stages; major crop in Indo-Gangetic Plain and Mediterranean regions |
| Maize | Highest global production cereal; strong phenology signal from satellite data; drought-sensitive during silking/grain-fill; major crop in US Corn Belt; contrasting growing season to wheat in many regions |

**Rationale for two crops:** Two crops provide sufficient diversity for testing cross-crop transfer (Claim C2) while remaining tractable. Wheat and maize have well-characterized phenology, extensive ground-truth data, contrasting stress sensitivities, and grow in different seasons/regions.

### 9.2 Study Region Selection

Three contrasting agro-climatic regions are selected based on data availability criteria specified in Section 6.3 of the Issue Resolution document:

| Region | Climate | Primary Crop | Data Justification |
|---|---|---|---|
| Indo-Gangetic Plain (Punjab/Haryana, India) | Subtropical monsoon; irrigated + rainfed | Wheat (rabi season) | ICRISAT district-level yield data; ISRO crop maps; known drought events (2002, 2009, 2014, 2022); high cloud-free observation density in rabi season (Oct-Mar); irrigation/rainfed contrast available |
| US Corn Belt (Iowa/Illinois) | Temperate continental; predominantly rainfed | Maize | USDA NASS county-level yield data; CDL crop type maps at 30m; extensive weather station network; known drought events (2012, 2023); low cloud contamination in growing season |
| Iberian Peninsula (Castilla-y-León/Andalucía, Spain) | Mediterranean; water-limited; drought-prone | Wheat | Eurostat/CAP yield data; ESA WorldCereal; documented severe drought events (2017, 2019, 2022-2023); strong irrigation/rainfed contrast; tests model under water-limited conditions |

**Region selection criteria verification:**

| Criterion | IGP | US Corn Belt | Iberian Peninsula |
|---|---|---|---|
| Crop map available | Yes (ISRO/Gov India) | Yes (CDL 30m) | Yes (ESA WorldCereal) |
| Sufficient cloud-free observations | Yes (rabi season) | Yes (summer) | Yes (spring) |
| Weather data for all years | Yes (ERA5/CHIRPS) | Yes (ERA5/NOAA) | Yes (ERA5) |
| Yield labels available | Yes (ICRISAT district) | Yes (USDA NASS county) | Yes (Eurostat/CAP) |
| Known drought events | 2002, 2009, 2014, 2022 | 2012, 2023 | 2017, 2019, 2022-2023 |
| Irrigation information | Yes (GMIA/IWMI) | Yes (USDA FRIS) | Yes (GMIA/IWMI) |
| 5+ years of data | Yes (2017-2023) | Yes (2017-2023) | Yes (2017-2023) |

### 9.3 Temporal Coverage

| Parameter | Specification |
|---|---|
| Study period | 2017-2023 (7 years) |
| Rationale | Post-Sentinel-2A/B launch (2015-2017 operational); captures multiple drought events; overlaps with HLS availability (from 2015); SMAP operational from 2015 |
| Growing seasons covered | ~7 seasons per crop per region |
| Temporal resolution | 5-day composites within growing season; monthly outside |
| Total time steps per season | ~24 five-day composites (120 days) for in-season modeling |

### 9.4 Train / Validation / Test Split Protocol

| Split | Method | Purpose | Data Fraction |
|---|---|---|---|
| Train | Random within-region, specific years | Model fitting | ~60% of years (2017-2020) |
| Validation | Out-of-year within-region | Hyperparameter selection, early stopping | ~20% of years (2021) |
| Test-In-Region | Out-of-year, different from validation | Final in-region performance | ~20% of years (2022) |
| Test-Out-Region | Entire held-out region | Spatial transfer evaluation | 1 region held out |
| Test-Extreme | Drought year holdout (2012 for US, 2022 for IGP) | Extreme-event robustness | Specific drought years |
| Test-Early-Season | Partial season observations only | No-leakage forecasting | Early-season subsets |

**Split locking rule:** All splits must be defined BEFORE model selection and training begins to prevent data leakage. Region/year/event splits are fixed and committed to version control.

**No-leakage protocol for forecasting tasks:**
- Future satellite observations beyond forecast date are PROHIBITED
- Full-season composites in early-season predictions are PROHIBITED
- Yield labels or statistics from the target prediction period are PROHIBITED
- Gap-filled products that use future observations are PROHIBITED unless explicitly documented
- Region-level normalization using held-out region statistics is PROHIBITED
- All preprocessing (normalization parameters, cloud thresholds) is fit only on training data

### 9.5 Label Sources and Scales

| Region | Crop | Label Source | Scale | Variables |
|---|---|---|---|---|
| IGP (India) | Wheat | ICRISAT VDSA | District-level | Yield (t/ha) |
| US Corn Belt | Maize | USDA NASS Quick Stats | County-level | Yield (bu/acre → t/ha) |
| Iberian Peninsula | Wheat | Eurostat/CAP | NUTS2/Province | Yield (t/ha) |

**Phenology pseudo-labels:** Derived from NDVI/EVI time-series derivatives (inflection points, local maxima/minima). These are noisy estimates, not ground truth. Label noise must be estimated and reported.

**Crop stress labels:** Defined as vegetation health anomalies below a threshold (e.g., NDVI anomaly < -1σ or VHI < 35). These are event-based labels validated against documented drought events.

**Scale compliance rule:** Predictions are validated ONLY at the spatial scale of the ground truth labels. If labels are district-level, predictions are aggregated to district level before computing yield metrics.

---

## 10. Data Processing Pipeline

### 10.1 Pipeline Stages

1. **Download and Archive:** Raw data download from APIs (Earth Engine, CDS, DAAC) with version control on download dates and product versions
2. **Coregistration:** Reproject to common CRS per region (UTM zones)
3. **Cloud/Quality Filtering:** Apply scene classification layer (SCL) for Sentinel-2; CFMask for Landsat; threshold: <20% cloud in 5-day composite
4. **5-day Compositing:** Median composite within 5-day windows; valid-pixel count recorded per pixel per composite
5. **Index Computation:** NDVI, EVI, NDWI, LAI, FAPAR computed from reflectance
6. **Temporal Alignment:** All data sources aligned to common 5-day time axis
7. **Multi-Resolution Stacking:** Data organized by resolution tier (fine/intermediate/coarse)
8. **Quality Flag Generation:** Per-pixel per-time-step quality flags (cloud, valid-observation count, gap-filled indicator, data source)
9. **Normalization:** Per-variable normalization using training-period statistics only
10. **Split Assignment:** Each pixel-time-step assigned to train/val/test split

### 10.2 Cloud Contamination Protocol

| Scenario | Strategy |
|---|---|
| 5-day composite has <20% cloud | Use optical data directly |
| 5-day composite has 20-80% cloud | Flag as partially cloud-contaminated; use with quality weight |
| 5-day composite has >80% cloud | Fall back to SAR/HLS/monthly composite; flag as gap-filled |
| Persistent cloud (>3 consecutive composites) | Use monthly aggregation; report as cloud-gap event |
| Tropical/monsoon regions with chronic cloud | Use SAR as primary; optical as secondary |

**Cloud-gap statistics:** Report valid-observation statistics by region and season as part of the Data Readiness Report.

---

## 11. Storage and Compute Estimates

### 11.1 Storage Estimates

| Data Category | Per Region (7 years) | 3 Regions Total |
|---|---|---|
| Sentinel-2 L2A (10m, 10 bands, 5-day) | ~2 TB | ~6 TB |
| HLS (30m, multi-band, 5-day) | ~500 GB | ~1.5 TB |
| ERA5 climate (daily, multi-variable) | ~50 GB | ~150 GB |
| CHIRPS precipitation | ~10 GB | ~30 GB |
| SoilGrids (static) | ~20 GB | ~60 GB |
| SMAP soil moisture | ~30 GB | ~90 GB |
| Crop masks and calendars | ~5 GB | ~15 GB |
| Yield labels and auxiliary | ~1 GB | ~3 GB |
| Processed datacubes | ~1 TB | ~3 TB |
| **Total** | **~3.6 TB** | **~10.8 TB** |

### 11.2 Compute Estimates

| Task | Estimated GPU-Hours | Hardware |
|---|---|---|
| Datacube preprocessing | 200-500 CPU-hours | Multi-core CPU |
| Baseline training (RF/XGBoost) | 50-100 CPU-hours | CPU |
| Baseline training (LSTM/TCN) | 100-300 GPU-hours | 1× A100 |
| SSL pretraining (MAE/future prediction) | 1,000-3,000 GPU-hours | 4-8× A100 |
| Full model training (all SSL objectives) | 3,000-8,000 GPU-hours | 8× A100 |
| Physics regularization experiments | 500-1,000 GPU-hours | 4-8× A100 |
| Ablation experiments | 1,000-2,000 GPU-hours | 4-8× A100 |
| Transfer and robustness evaluation | 500-1,000 GPU-hours | 4-8× A100 |
| **Total estimated** | **6,000-15,000 GPU-hours** | **8× A100 (80GB)** |

**Minimum compute requirement:** Access to at least one 8×A100 node for ~3 months of active training.

### 11.3 Reproducibility Requirements

| Requirement | Specification |
|---|---|
| Random seeds | Fixed seeds for all experiments; logged in config files |
| Software versions | PyTorch, TorchGeo, xarray, rasterio versions frozen |
| Data download scripts | Version-controlled scripts with product IDs and dates |
| Train/val/test split files | Committed to repository before any model training |
| Model configuration files | YAML/JSON configs for every experiment |
| Experiment tracking | Weights & Biases or MLflow for all training runs |
| Preprocessing pipeline | Fit on training data only; saved transformers applied to val/test |

---

# PART IV: SELF-SUPERVISED LEARNING

---

## 12. SSL Objective Specifications

### 12.1 SSL Objective Sequence

Objectives are introduced progressively, NOT trained simultaneously from the start:

| Stage | Objective ID | Objective | Purpose | Activation |
|---|---|---|---|---|
| SSL-1 | L_ssl_1 | Masked Spatiotemporal Reconstruction | Learn basic EO representations | Phase 1 (Warm-up) |
| SSL-2 | L_ssl_2 | Cross-Modal Masked Prediction | Learn relationships among weather, soil, and vegetation | Phase 2 (Fusion) |
| SSL-3 | L_ssl_3 | Future Vegetation Trajectory Prediction | Learn temporal ecosystem dynamics | Phase 2-3 |
| SSL-4 | L_ssl_4 | Phenology Pseudo-Transition Prediction | Learn crop-stage timing | Phase 3 (Process) |
| SSL-5 | L_ssl_5 | Drought Stress Trajectory Prediction | Learn drought response dynamics | Phase 3 (Process) |
| Scenario-1 | L_scenario | Physics-Constrained Scenario Augmentation | Scenario consistency regularization | Phase 5 (Post-physics) |

### 12.2 Detailed Objective Descriptions

#### SSL-1: Masked Spatiotemporal Reconstruction

**Objective:** Randomly mask patches in satellite imagery (both spatially and temporally) and train the model to reconstruct the masked regions.

**Masking strategy:**
- Spatial masking: 75% of spatial patches masked per time step (following MAE)
- Temporal masking: 20% of time steps fully masked
- Combined: Random spatial-temporal masking with varying ratios

**Loss:** L1 reconstruction loss on masked patches only

**Rationale:** This is the foundational SSL objective that learns to represent spatial and temporal structure in satellite imagery. High masking ratio forces the model to learn meaningful representations rather than simple interpolation.

#### SSL-2: Cross-Modal Masked Prediction

**Objective:** Mask one modality (e.g., weather) and predict it from remaining modalities (e.g., satellite + soil + crop calendar).

**Implementation:**
- Weather → Satellite prediction: Predict NDVI/EVI trajectory from weather + soil + crop calendar
- Satellite → Weather prediction: Predict temperature/precipitation from vegetation + soil
- Soil → Vegetation prediction: Predict vegetation response from soil + weather

**Loss:** L2 prediction loss on masked modality tokens

**Rationale:** Cross-modal prediction forces the model to learn the relationships between climate forcing, soil conditions, and vegetation response. This is directly aligned with learning the causal process chain.

#### SSL-3: Future Vegetation Trajectory Prediction

**Objective:** Given vegetation state up to time t, predict the vegetation trajectory from t+1 to t+K (K=3-6 five-day steps ahead).

**Input:** Satellite + weather + soil up to time t
**Target:** NDVI/EVI values for t+1 through t+K

**Loss:** Weighted L2 trajectory loss with increasing weight for near-future predictions

**Rationale:** This objective teaches temporal dynamics — the model must understand how vegetation evolves under current conditions to predict its future trajectory. This is essential for drought forecasting and phenology prediction.

#### SSL-4: Phenology Pseudo-Transition Prediction

**Objective:** Predict whether a phenology transition will occur within the next K time steps, and identify the transition type.

**Input:** Multimodal state up to time t
**Target:** Phenology transition labels derived from NDVI/EVI derivative analysis (green-up onset, peak, senescence onset, etc.)

**Label quality:** These are pseudo-labels derived from vegetation index derivatives, NOT ground truth. Label noise must be estimated.

**Loss:** Cross-entropy for transition classification + regression for transition timing

**Rationale:** Directly teaches the model to recognize and predict phenology transitions, which are the key temporal events in the crop development cycle.

#### SSL-5: Drought Stress Trajectory Prediction

**Objective:** Given current multimodal state, predict the drought stress trajectory (VHI/NDVI anomaly evolution) over the next K time steps.

**Input:** Multimodal state up to time t
**Target:** Drought index values (VHI, SPEI-category) for t+1 through t+K

**Loss:** L2 trajectory loss + classification loss for drought onset/severity

**Rationale:** Teaches the model to understand drought evolution dynamics, which is the core drought-modeling capability.

#### Scenario-1: Physics-Constrained Scenario Augmentation (NOT strict SSL)

**Objective:** Apply synthetic perturbations (e.g., reduce rainfall by 30%, increase temperature by 2°C) and enforce that the model's predicted response is physically consistent (e.g., reduced soil moisture, increased ET, decreased vegetation health).

**Status:** This is reclassified from "Counterfactual SSL" to "Physics-Constrained Scenario Augmentation" because synthetic interventions encode designer assumptions, not real-world counterfactual behavior. It is a consistency regularization technique, not a self-supervised learning objective.

**Activation:** Only after physics regularization is stable (Phase 5)

**Loss:** Consistency loss between predicted response and physics-expected response under synthetic perturbation

**Rationale:** Enforces that the model's representations respond to environmental perturbations in a physically plausible manner, providing a form of physics-guided regularization.

---

## 13. Loss Function Specification

### 13.1 Total Training Loss

```
L_total = Σ_i α_i × L_SSL_i
        + λ_water(t) × L_water
        + λ_growth(t) × L_growth
        + λ_causal(t) × L_temporal_DAG
        + λ_calib × L_calibration
```

Where:
- `α_i` (i=1..5): SSL objective weights
- `λ_water(t)`: Water-balance physics loss weight with curriculum scheduling
- `λ_growth(t)`: Plant-growth physics loss weight with curriculum scheduling
- `λ_causal(t)`: Temporal DAG consistency loss weight with curriculum scheduling
- `λ_calib`: Calibration loss weight (constant)

### 13.2 SSL Objective Weight Ranges

| Weight | Initial Value | Range | Schedule |
|---|---|---|---|
| α_1 (Reconstruction) | 1.0 | 0.5-2.0 | High initially, decrease as other objectives activate |
| α_2 (Cross-Modal) | 0.5 | 0.3-1.0 | Start at 0, ramp to 0.5 in Phase 2 |
| α_3 (Future Trajectory) | 0.8 | 0.5-1.5 | Start at 0, ramp to 0.8 in Phase 2-3 |
| α_4 (Phenology) | 0.5 | 0.3-1.0 | Start at 0, ramp to 0.5 in Phase 3 |
| α_5 (Drought) | 0.5 | 0.3-1.0 | Start at 0, ramp to 0.5 in Phase 3 |

### 13.3 Physics Loss Weight Warm-up Schedules

| Weight | Warm-up Period | Start Value | Target Value | Schedule |
|---|---|---|---|---|
| λ_water(t) | 5 epochs | 0.0 | 0.1 | Linear warm-up |
| λ_growth(t) | 5 epochs (after λ_water) | 0.0 | 0.05 | Linear warm-up |
| λ_causal(t) | 5 epochs (after λ_growth) | 0.0 | 0.05 | Linear warm-up |
| λ_calib | No warm-up | 0.01 | 0.01 | Constant |

**Principle:** Physics and causal losses are introduced ONLY after the SSL objectives have stabilized. Their weights start at zero and increase gradually to avoid destabilizing the learned representations.

### 13.4 Training Phase Schedule

| Phase | Duration (approx.) | Active Losses | Focus |
|---|---|---|---|
| Warm-up | Epochs 1-20 | L_ssl_1 (reconstruction) | Learn basic EO representations |
| Fusion | Epochs 21-50 | L_ssl_1 + L_ssl_2 + L_ssl_3 | Learn multimodal and temporal relationships |
| Process | Epochs 51-80 | + L_ssl_4 + L_ssl_5 | Learn phenology and drought dynamics |
| Physics | Epochs 81-100 | + λ_water × L_water + λ_growth × L_growth | Physics regularization |
| Causal-informed | Epochs 101-120 | + λ_causal × L_temporal_DAG | Causal consistency (only if previous phases pass gates) |

### 13.5 Gradient Monitoring

| Monitor | Threshold | Action |
|---|---|---|
| Total gradient norm | >100 | Reduce learning rate |
| Per-objective gradient norm ratio | >10:1 between any two objectives | Rebalance weights |
| Gradient cosine similarity between objectives | <0 (conflicting) | Reduce weight of conflicting objective |
| Loss scale divergence | Any loss >10× initial | Pause that objective; reduce weight |

---

## 14. SSL Stability Monitoring

### 14.1 Monitored Quantities

| Metric | Purpose | Collapse Indicator |
|---|---|---|
| Embedding variance | Detect dimensional collapse | Variance → 0 |
| Embedding rank | Detect rank collapse | Rank → 1 |
| Covariance spectrum | Detect spectral collapse | Single dominant eigenvalue |
| Nearest-neighbor diversity | Detect representation clustering | All samples map to same centroid |
| Objective-specific validation losses | Detect overfitting or divergence | Val loss diverges from train |
| Gradient norms | Detect training instability | Exploding or vanishing gradients |
| Linear-probe performance | Detect representation quality over training | Probe accuracy does not improve |

### 14.2 Collapse Response Protocol

| Condition | Response |
|---|---|
| Embedding variance < threshold | Stop training; increase temperature in contrastive losses; add noise |
| Rank collapse detected | Reduce model capacity; add spectral regularization |
| Single objective degrades all downstream tasks | Remove or downweight that objective |
| Negative transfer between SSL tasks | Train single-objective ablations; identify conflicting pair |
| SSL provides no benefit over supervised-from-scratch | Revert to supervised training; report SSL as negative finding |

---

# PART V: PHYSICS FRAMEWORK

---

## 15. Water-Balance Constraint

### 15.1 Revised Water-Balance Equation

```
ΔSM = P + I_latent - ET - R - D + ε
```

| Symbol | Meaning | Units | Source |
|---|---|---|---|
| ΔSM | Change in soil moisture | mm | SMAP/ESA CCI (observed) |
| P | Precipitation | mm/5-day | ERA5/CHIRPS |
| I_latent | Latent unobserved water input | mm/5-day | Inferred by model (NOT direct input) |
| ET | Evapotranspiration | mm/5-day | ERA5 ET0 × crop coefficient; or remote sensing ET product |
| R | Runoff | mm/5-day | Estimated from rainfall intensity + soil properties |
| D | Drainage / deep percolation | mm/5-day | Estimated from soil moisture surplus + soil hydraulic properties |
| ε | Measurement/model residual | mm/5-day | Remaining unexplained variation |

### 15.2 I_latent: Latent Residual Water-Input Inference

**Definition:** I_latent is a model-inferred variable representing all unobserved water inputs that cannot be explained by measured precipitation alone. It is NOT "irrigation detection."

**What I_latent may capture:**
- Irrigation
- Groundwater contribution to root zone
- Capillary rise from shallow water table
- Runoff model error (lateral inflow not captured)
- Rainfall data error (undercatch, spatial interpolation error)
- Soil moisture retrieval error
- Cloud/gap-filling artifact
- Crop-rooting-depth effects (deeper roots accessing moisture)

**Validation of I_latent:** Compare residual anomalies with known irrigation maps, canal command areas, and reported irrigation districts where available. Only after validation against multiple evidence sources should I_latent be interpreted as irrigation.

**I_latent acceptance criteria:**
- Residual variable improves error analysis (identifies where water balance fails)
- Residual variable does NOT degrade transfer performance
- Residual anomalies are spatially and temporally structured (not random noise)
- Residual anomalies correlate with known irrigation districts where data available

### 15.3 Physics Loss Design

The physics loss is uncertainty-weighted:

```
L_water = w_data × ||ΔSM_pred - (P + I_latent - ET - R - D)||²
```

Where `w_data` is an uncertainty weight that is:
- Higher when data quality is good (low missingness, fine resolution)
- Lower when data quality is poor (high missingness, coarse resolution, known biases)
- Zero when data is entirely missing

The model predicts ΔSM_pred and I_latent jointly. The physics loss measures consistency between the predicted soil moisture change and the water-balance equation.

### 15.4 Scale Considerations for Water Balance

The water-balance equation is most valid at the scale at which its components are measured. At 10m, rainfall, ET, and soil moisture products are too coarse to close the water balance reliably. The physics loss should be computed at the coarsest-resolution tier (1-10km) or at the natural aggregation scale of the dominant forcing data.

**Implementation:** Compute physics loss on coarse-resolution tokens (weather/soil-moisture scale), NOT on fine-resolution satellite tokens. Fine-resolution predictions are evaluated for task performance but are not directly constrained by pixel-level water balance.

---

## 16. Plant-Growth Constraint

### 16.1 Growing Degree Days Accumulation

```
GDD_t = max(0, T_mean_t - T_base)
AGDD_t = Σ GDD_τ for τ from planting date to t
```

| Parameter | Wheat | Maize |
|---|---|---|
| T_base | 0°C | 10°C |
| T_upper (optional) | 30°C | 30°C |
| Typical AGDD to maturity | 1500-2000 °C·day | 1200-1800 °C·day |

### 16.2 Growth Constraint Formulation

The plant-growth constraint enforces that predicted vegetation health and phenology progression are consistent with accumulated thermal time and water availability:

```
L_growth = w_growth × (
    ||predicted_VHI - f(AGDD, water_availability)||²
    + ||predicted_phenology_stage - g(AGDD, stress_accumulation)||²
)
```

Where:
- `f(AGDD, water_availability)` is a simplified Mitscherlich-Baule growth response function
- `g(AGDD, stress_accumulation)` enforces that phenology cannot advance without sufficient accumulated thermal time
- Stress accumulation can delay phenology but cannot advance it beyond thermal time limits

### 16.3 Plant Growth Validation Metrics

| Metric | Purpose | Acceptance |
|---|---|---|
| Impossible-state rate | Count predictions violating basic physical logic (e.g., green-up in winter, senescence during peak AGDD) | <1% of predictions |
| Phenology-AGDD consistency | Check phenology predictions are consistent with accumulated GDD | Correlation >0.9 between predicted phenology timing and AGDD thresholds |
| Stress-under-deficit check | Healthy growth should not be predicted under severe deficit without I_latent explanation | Flag violations; document |

---

## 17. Physics Acceptance Criteria

The physics-regularized model is accepted ONLY if ALL of the following conditions are met:

1. **Water-balance residual improves:** RMSE of water-balance residual decreases by ≥15% relative to the non-physics model
2. **Task performance does not collapse:** Primary task metrics (crop stress AUROC, phenology MAE, yield RMSE) do not degrade by more than 3% relative to the non-physics model
3. **Transfer improves or remains stable:** Out-of-region performance does not degrade with physics
4. **Residuals are interpretable:** Water-balance residual anomalies show spatial and temporal structure, not random noise; they correlate with known irrigation districts where data available

**Fallback path:** If physics fails acceptance criteria, physics will be reported as a diagnostic analysis rather than a core model contribution. The paper will document the physics experiment as a negative or inconclusive finding.

---

# PART VI: CAUSAL FRAMEWORK

---

## 18. Temporal DAG Specification

### 18.1 Complete Temporal DAG Equations

```
Equation 1: Weather(t) → ET(t)
  ET(t) = f_ET(Temperature(t), Solar_Radiation(t), Wind(t), Humidity(t))

Equation 2: Rainfall(t), ET(t), Soil(t), I_latent(t) → Soil_Moisture(t)
  SM(t) = SM(t-1) + P(t) + I_latent(t) - ET(t) - R(t) - D(t) + ε(t)

Equation 3: Soil_Moisture(t), Temperature(t), Crop_Stage(t) → Vegetation_Health(t)
  VH(t) = f_VH(SM(t), Temperature(t), Crop_Stage(t), Stress_Sensitivity(Crop_Stage(t)))

Equation 4: Vegetation_Health(t), AGDD(t), Crop_Calendar → Crop_Stage(t+1)
  CS(t+1) = f_CS(VH(t), AGDD(t), Crop_Calendar_Prior(t+1))

Equation 5: Vegetation_Health(t), Crop_Stage(t), Stress_Accumulation(t) → Yield_Risk(t+1)
  YR(t+1) = f_YR(Σ w(CS(τ)) × Stress(τ) for τ = planting to t)
```

### 18.2 Edge Strength Learning

In the first version, the causal graph structure is domain-specified (fixed topology), and only edge strengths are learnable. Full neural causal discovery is deferred to Phase 2.

| Edge | Domain-Fixed Direction | Learnable Parameter |
|---|---|---|
| Weather → ET | Fixed | Strength coefficient |
| Rainfall → Soil Moisture | Fixed | Strength coefficient |
| ET → Soil Moisture | Fixed | Strength coefficient |
| I_latent → Soil Moisture | Fixed | Strength coefficient |
| Soil Moisture → Vegetation Health | Fixed | Strength coefficient |
| Temperature → Vegetation Health | Fixed | Strength coefficient |
| Crop Stage → Vegetation Health | Fixed | Strength coefficient |
| Vegetation Health → Crop Stage (t+1) | Fixed | Strength coefficient |
| AGDD → Crop Stage (t+1) | Fixed | Strength coefficient |
| Vegetation Health → Yield Risk (t+1) | Fixed | Strength coefficient |
| Crop Stage → Yield Risk (t+1) | Fixed | Strength coefficient |
| Stress Accumulation → Yield Risk (t+1) | Fixed | Strength coefficient |

### 18.3 Causal Loss

```
L_temporal_DAG = λ_causal(t) × [
    Σ_edges ||predicted_effect - learned_strength × cause||
    + acyclicity_penalty
    + sign_consistency_penalty
]
```

The acyclicity penalty ensures no cycles exist in the temporal unrolled graph. The sign consistency penalty enforces domain-expected directional effects (e.g., more rainfall should not decrease soil moisture under normal conditions).

---

## 19. Causal Validation Metrics

### 19.1 Causal Validation Tests

| Test | Purpose | Acceptance | Failure Action |
|---|---|---|---|
| Graph acyclicity after temporal unrolling | Ensures structural validity | No directed cycles | Redesign temporal DAG |
| Directional sign checks | Rainfall should increase SM; ET should decrease SM; Temperature increase should increase stress | >95% of test samples show correct signs | Debug causal layer; check data processing |
| Invariance across years | Stable edge strengths across different years | Coefficient of variation <0.3 for key edges | Reduce causal claims; report instability |
| Invariance across regions | Stable causal relationships across regions | Same sign and similar magnitude | Report region-specific behavior |
| Extreme event behavior | Model captures known drought events | Correct stress signal in documented drought years | Inspect I_latent; check data |
| Irrigated vs rainfed contrast | I_latent should be higher in irrigated areas | Statistically significant difference (t-test, p<0.05) | Reconsider I_latent interpretation |
| Scenario monotonicity | Less rainfall should generally not produce lower stress (unless I_latent compensates) | Monotonic response in >90% of scenarios | Document exceptions; check confounders |
| Known-event case studies | Compare model explanations against documented events | Qualitative agreement with field reports | Document disagreement |

### 19.2 Causal Terminology Guidelines

| Allowed Terminology | Prohibited Terminology (in MVRC) |
|---|---|
| Causally informed structure | Causal foundation model |
| Causally structured scenario analysis | Fully causal decision engine |
| Temporal DAG-guided representation | Proven causal discovery |
| Scenario simulation | Counterfactual inference |
| Causal plausibility checks | Validated causal effect estimates |

---

## 20. Counterfactual Module Status

### 20.1 Phased Counterfactual Plan

| Stage | Status | Scope | Validation Requirement |
|---|---|---|---|
| Year 1 (MVRC) | Limited scenario-sensitivity tests only | Apply synthetic perturbations; check directional responses | Sign checks, monotonicity |
| Year 2 (Phase 2) | Formal counterfactual module | If temporal-DAG and physics pass validation | Intervention-like evidence; natural experiment validation |

### 20.2 Counterfactual Output Labeling

All Year 1 counterfactual outputs must be labeled as **"scenario simulations"** unless validated using intervention-like evidence. The term "counterfactual inference" is reserved for Phase 2 and requires:
- Validated temporal DAG (G6 passed)
- Stable physics constraints (G5 passed)
- At least one natural experiment or quasi-experimental validation
- Sign, magnitude, and monotonicity checks passed

---

# PART VII: EVALUATION FRAMEWORK

---

## 21. Baseline Specifications

### 21.1 Required Baselines

| # | Baseline | Purpose | Implementation |
|---|---|---|---|
| 1 | Climatology / Historical Mean | Minimum skill baseline | Average conditions over training period |
| 2 | Persistence Model | Temporal forecasting baseline | Predict current state continues |
| 3 | GDD Phenology Model | Simple crop-physics baseline | Accumulate GDD; predict phenology from thermal thresholds |
| 4 | SPI/SPEI/VHI Drought Index Models | Drought-monitoring baselines | Standard drought indices from climate data |
| 5 | Random Forest / XGBoost | Strong tabular engineered-feature baseline | Hand-crafted features: NDVI trajectory, weather stats, soil properties, crop calendar |
| 6 | LSTM / TCN | Deep temporal baseline | Temporal deep learning on time-series features |
| 7 | Satellite-only Model | Tests value of imagery alone | Same architecture but only satellite encoder |
| 8 | Weather-only Model | Tests value of climate data alone | Same architecture but only weather encoder |
| 9 | Multimodal No-Physics Model | Tests physics contribution | Full multimodal fusion without physics losses |
| 10 | Multimodal No-Process-SSL Model | Tests process SSL contribution | Generic MAE SSL instead of process-centered SSL |
| 11 | Existing GeoFM Features (if feasible) | Tests value over general EO representations | SatMAE/DOFA/PhilEO features + linear probe |
| 12 | AquaCrop/DSSAT/APSIM (optional) | Process-model comparison | Only if detailed management data available |

### 21.2 Baseline Priority

| Priority | Baselines | Must-include? |
|---|---|---|
| Critical | 1, 2, 5, 6, 9, 10 | Yes |
| Important | 3, 4, 7, 8 | Yes |
| Valuable | 11, 12 | If feasible |

---

## 22. Evaluation Split Protocol

| Split ID | Split Type | Purpose | Data Used |
|---|---|---|---|
| S1 | Random within-region | Basic sanity check (not final proof) | Random pixel-level split within training years |
| S2 | Out-of-year | Temporal generalization | Year 2022 held out from training |
| S3 | Out-of-region | Spatial transfer | One region entirely held out |
| S4 | Extreme-event holdout | Drought/heat robustness | Documented drought years held out |
| S5 | Early-season prediction | No-leakage forecasting | Only first 40/60/80 days of season |
| S6 | Irrigated/rainfed contrast (if available) | Confounding analysis | Split by irrigation status |

---

## 23. Task Metrics and Thresholds

### 23.1 Crop Stress Detection

| Metric | Description | Minimum Threshold | Target |
|---|---|---|---|
| AUROC | Area under ROC curve | >0.80 | >0.90 |
| AUPRC | Area under Precision-Recall curve | >0.60 | >0.80 |
| Event-level F1 | F1 for drought event detection | >0.50 | >0.70 |
| Spatial IoU | Intersection over Union of stress maps | >0.30 | >0.50 |

### 23.2 Phenology Forecasting

| Metric | Description | Minimum Threshold | Target |
|---|---|---|---|
| MAE (days) | Mean absolute error of transition timing | <10 days | <5 days |
| Transition F1 | F1 score for detecting phenology transitions | >0.60 | >0.80 |
| Stage accuracy | Accuracy of phenology stage classification | >0.70 | >0.85 |
| Calibration | ECE (Expected Calibration Error) | <0.10 | <0.05 |

### 23.3 Drought Forecasting

| Metric | Description | Minimum Threshold | Target |
|---|---|---|---|
| RMSE (drought index) | Root mean square error of predicted index | <0.5 (standardized) | <0.3 |
| Onset F1 | F1 for drought onset detection | >0.50 | >0.70 |
| Lead-time skill score | Skill relative to climatology at various lead times | Positive at 20-day lead | Positive at 40-day lead |

### 23.4 Yield-Risk Prediction

| Metric | Description | Minimum Threshold | Target |
|---|---|---|---|
| RMSE (yield) | Root mean square error | <15% of mean yield | <10% |
| MAE (yield) | Mean absolute error | <12% of mean yield | <8% |
| R² | Coefficient of determination | >0.50 | >0.70 |
| AUC (below-threshold risk) | AUC for detecting yield below threshold | >0.75 | >0.85 |

### 23.5 Representation Quality

| Metric | Description | Minimum Threshold |
|---|---|---|
| Linear probing accuracy | Frozen representation + linear classifier | >supervised-from-scratch baseline |
| Fine-tuning efficiency | Performance vs. number of fine-tuning samples | Better than random-init at <50% data |
| Transfer gap | Difference in fine-tuning performance across regions | Smaller than supervised baseline |

### 23.6 Physics Consistency

| Metric | Description | Acceptance Criterion |
|---|---|---|
| Water-balance residual RMSE | Physical consistency measure | ≥15% improvement over non-physics model |
| Impossible-state rate | Predictions violating basic physics | <1% |
| Stress-under-deficit check | Healthy growth under severe deficit without I_latent explanation | <5% violation rate |

### 23.7 Causal Plausibility

| Metric | Description | Acceptance Criterion |
|---|---|---|
| Sign correctness | Correct directional effects | >95% of test cases |
| Monotonicity | Plausible response to interventions | >90% of scenarios |
| Invariance (cross-year) | Stable edge strengths | CV <0.3 for key edges |
| Known-event agreement | Qualitative match with documented events | Documented case studies agree |

### 23.8 Calibration

| Metric | Description | Minimum Threshold |
|---|---|---|
| Brier score | Probabilistic calibration | <0.2 for binary tasks |
| ECE | Expected Calibration Error | <0.10 |
| Reliability curves | Visual calibration check | Close to diagonal |
| Prediction interval coverage | % of true values within prediction intervals | Within 5% of nominal coverage |

---

# PART VIII: IMPLEMENTATION ROADMAP

---

## 24. 52-Week MVRC Roadmap

### 24.1 Phase Breakdown

| Weeks | Phase | Output | Gate |
|---|---|---|---|
| 1-2 | Revision and scope lock | Revised hypothesis, MVRC definition, study regions, tasks, metrics, claim register | G0 |
| 3-8 | Data audit and datacube prototype | Multi-resolution datacube for one pilot region; Data Readiness Report | G1 |
| 9-12 | Domain and tool strengthening | Agronomy/drought/phenology knowledge track completed | — |
| 13-18 | Baselines | Climatology, persistence, XGBoost/RF, GDD/SPI/VHI, LSTM/TCN results | G2 |
| 19-26 | MVP model | Satellite/weather/soil/crop encoders and fusion module working | G3 |
| 27-34 | Process-centered SSL | Future trajectory, phenology, drought objectives implemented and evaluated | G4 |
| 35-40 | Physics regularization | Water-balance and growth-loss ablations completed | G5 |
| 41-45 | Transfer and robustness testing | Out-of-region/year/event evaluation completed | — |
| 46-49 | Causal-informed diagnostics | Temporal-DAG sign/invariance/scenario tests | G6 |
| 50-52 | Final integration and report | Ablation tables, error analysis, manuscript-ready results | G7 |

### 24.2 24-Month Full Vision (Phase 2 Extension)

| Months | Scope |
|---|---|
| 13-18 | Full causal discovery with learnable graph structure |
| 13-18 | Formal counterfactual module with intervention-style validation |
| 15-20 | Additional crop types and regions |
| 18-24 | Food-security risk layer with socioeconomic data |
| 18-24 | DSSAT/APSIM/AquaCrop hybrid integration |
| 20-24 | Global multi-biome generalization study |
| 22-24 | Operational deployment prototype |

---

## 25. Implementation Gates

### 25.1 Gate Definitions

| Gate | Name | Go Condition | No-Go Condition | Decision |
|---|---|---|---|---|
| G0 | Scope Lock | Final crop/region/year/task/metric/data selection is testable and data exists | No available data for any proposed region-crop combination | Redefine scope; select different regions |
| G1 | Data Readiness | At least 2 regions have sufficient satellite/weather/soil/crop/yield data; cloud/missingness profile acceptable | No reliable labels or severe missingness with no fallback | Fix data pipeline; select alternative regions |
| G2 | Baseline Performance | At least one simple baseline beats climatology/persistence; evaluation pipeline stable | Labels or task definition appear invalid | Revisit task formulation; check labels |
| G3 | Multimodal Fusion Value | Fusion improves over best single-modality baseline in at least 1 primary task or transfer split | Fusion adds no value after alignment/missingness fixes | Analyze attention patterns; try alternative fusion |
| G4 | Process SSL Value | Process SSL improves transfer, fine-tuning efficiency, or extreme-event performance | SSL collapses or provides no measurable benefit | Revert to generic SSL; report as negative finding |
| G5 | Physics Value | Physics improves consistency (≥15% residual reduction) without major task degradation (≤3% drop) | Physics worsens both residuals and task metrics | Report physics as diagnostic only |
| G6 | Causal-Informed Value | Temporal-DAG passes sign, monotonicity, and invariance checks | Scenario behavior is unstable or contradicts domain knowledge | Treat causal module as interpretability only |
| G7 | Research Readiness | Ablations support at least 1 clear contribution; results reproducible; failure modes documented | Results are broad but inconclusive | Extend analysis; target specific contribution |

---

## 26. Compute and Reproducibility Plan

### 26.1 Compute Budget

| Item | Specification |
|---|---|
| Hardware | 8× NVIDIA A100 (80GB) GPU node |
| Minimum access period | 3 months of active training |
| Estimated total GPU-hours | 6,000-15,000 GPU-hours |
| Data storage | ~11 TB for 3 regions |
| Preprocessing compute | 200-500 CPU-hours |
| Estimated cloud cost (if applicable) | $15,000-40,000 (AWS p4d.24xl equivalent) |

### 26.2 Reproducibility Checklist

| Item | Status |
|---|---|
| Fixed random seeds for all experiments | To be set before training |
| Software version freeze (PyTorch, TorchGeo, xarray, rasterio) | To be documented |
| Data download scripts with product IDs and dates | To be version-controlled |
| Train/val/test split files | To be committed BEFORE any model training |
| Model configuration files (YAML/JSON) | To be version-controlled |
| Experiment tracking (W&B/MLflow) | To be configured |
| Preprocessing pipeline (fit on train only) | To be implemented with saved scalers |
| Every final table reproducible from config + splits | To be verified |

---

# PART IX: RISK AND MITIGATION

---

## 27. Risk Register

| # | Risk | Severity | Why It Matters | Mitigation | Fallback |
|---|---|---|---|---|---|
| R1 | Overbroad foundation-model scope | High | Could prevent completion | MVRC defined for 52 weeks | Narrow scope further; defer modules |
| R2 | False 10m precision | High | Invalid physical and causal claims | Multi-resolution datacube adopted | Aggregate all outputs to coarser validation scale |
| R3 | Weak ground truth | High | Evaluation may become inconclusive | Select regions based on label availability first | Use proxy labels with documented uncertainty |
| R4 | Causal overclaiming | High | Peer-review vulnerability | Temporal DAG; invariance tests; cautious terminology | Treat causal module as interpretability only |
| R5 | Missing irrigation data | High | Confounds drought-stress/yield relationships | I_latent latent residual strategy | Treat irrigation as uncertainty/confounder |
| R6 | Physics loss misspecification | Medium-High | Can degrade performance | Uncertainty-weighted soft constraints | Report physics as diagnostic only |
| R7 | Compute/storage underestimation | Medium-High | Training may stall | Define compute budget before architecture lock | Reduce model size; fewer ablations |
| R8 | Multimodal alignment errors | Medium-High | Model may learn artifacts | Strict data QA and visual checks | Reduce number of modalities in MVP |
| R9 | Too many SSL objectives | Medium | Debugging becomes impossible | Add objectives sequentially | Train objectives separately |
| R10 | Food-security overreach | Medium | Requires socioeconomic data | Reframed as agricultural production risk | Remove food-security claims entirely |
| R11 | SSL collapse | Medium | Representation quality fails | Embedding variance/rank monitoring | Restart with adjusted hyperparameters |
| R12 | Negative transfer between SSL tasks | Medium | Joint training harms some tasks | Single-objective ablations | Remove harmful objective |
| R13 | Cloud contamination in tropical regions | Medium | Insufficient valid observations | SAR fallback; monthly composites | Exclude worst-affected regions/seasons |
| R14 | Temporal lag ambiguity | Medium | Weather-vegetation lag unclear | Lag-window ablations | Select lag by validation performance |
| R15 | Crop-mask noise | Medium | Wrong crop labels contaminate learning | Confidence filtering; sensitivity testing | Use high-confidence pixels only |
| R16 | Yield-label scale mismatch | Medium | Cannot validate field-level from admin labels | Aggregate predictions to label scale | Validate only at label scale |
| R17 | Calibration missing | Medium | Overconfident predictions | ECE, Brier score, reliability curves | Add calibration post-hoc |
| R18 | Domain knowledge gap | Medium | Researcher may miss agronomic nuance | Structured reading track | Consult domain experts |

---

## 28. Fallback Paths Summary

| Scenario | Fallback Action |
|---|---|
| Physics harms task performance | Report physics as diagnostic; claim only task performance |
| Causal module fails sign/monotonicity checks | Treat as interpretability tool; remove causal claims |
| SSL provides no benefit over supervised | Report supervised results; document SSL as negative finding |
| Fusion adds no value over single modality | Report single-modality results; analyze failure mode |
| I_latent is uninterpretable | Remove I_latent; treat irrigation as documented confounder |
| Multi-resolution fusion too complex | Fall back to moderate-resolution (250m-1km) single-grid approach |
| Drought events insufficient in data | Extend temporal window; include earlier satellite data (Landsat-only era) |
| Compute budget exceeded | Reduce model size; fewer ablations; single-region study |

---

## 29. Ethical and Communication Constraints

### 29.1 Allowed Language

- Agricultural production-risk prototype
- Crop-stress monitoring model
- Yield-shortfall risk estimator
- Causally informed scenario analysis
- Multimodal representation learning framework

### 29.2 Prohibited Language (in MVRC/Year 1)

- Food-security crisis prediction
- Fully causal decision engine
- Guaranteed climate adaptation recommendation
- Field-level yield truth (if labels are administrative)
- Operational decision-support system
- Validated counterfactual inference

### 29.3 Required Disclaimers

All maps and risk outputs must include:
- Uncertainty estimates (prediction intervals or calibrated probabilities)
- Caveats about spatial scale limitations
- Statement that the model is a research prototype, not an operational system
- Acknowledgment of data limitations (cloud gaps, coarse resolution, missing irrigation)

---

# PART X: VALIDATION HISTORY AND CORRECTIONS

---

## 30. Blocker Resolutions (Resolved)

### 30.1 B1: DAG Cycle — RESOLVED

| Aspect | Detail |
|---|---|
| Original problem | Bidirectional edge VH ↔ CS at same time step violated DAG acyclicity |
| Resolution | Temporal DAG: VH(t) → CS(t+1) and CS(t) → VH(t) (no same-time bidirectionality) |
| Validation | Check graph acyclicity after temporal unrolling; no same-time directed cycles |
| Fallback | Remove causal module from MVRC; keep interpretability only |

### 30.2 B2: Timeline Realism — RESOLVED

| Aspect | Detail |
|---|---|
| Original problem | 52 weeks insufficient for full plan; realistic timeline 70-80 weeks |
| Resolution | Define MVRC (Minimum Viable Research Contribution) for 52 weeks; full plan as 24-month vision |
| Validation | Milestone review every 8 weeks; at least MVRC modules completed by Week 52 |
| Fallback | Convert causal/counterfactual to future work section |

### 30.3 B3: Irrigation Data Gap — RESOLVED

| Aspect | Detail |
|---|---|
| Original problem | No reliable global irrigation data at fine scales |
| Resolution | I_latent: latent residual water-input inference term (NOT "irrigation detection") |
| Validation | Compare residual anomalies with irrigation maps where available; structured (not random) residuals |
| Fallback | Treat irrigation as documented confounder/uncertainty source |

### 30.4 B4: Food-Security Overreach — RESOLVED

| Aspect | Detail |
|---|---|
| Original problem | Food-security risk requires socioeconomic data not included in model |
| Resolution | Reframed as "agricultural production risk" / "yield-shortfall risk" |
| Validation | No food-security claims without socioeconomic data |
| Fallback | Add food-security layer only in Phase 2 extension |

---

## 31. High-Priority Resolutions (Resolved)

### 31.1 H1: Forced 10m Data Harmonization — RESOLVED

| Aspect | Detail |
|---|---|
| Problem | Resampling coarse data to 10m creates false precision |
| Resolution | Multi-resolution datacube with scale-aware fusion |
| Validation | No coarse variables interpreted as true 10m signals |

### 31.2 H2: Loss Weighting Unspecified — RESOLVED

| Aspect | Detail |
|---|---|
| Problem | Multi-objective loss with no specified weights or curriculum |
| Resolution | Weighted loss with curriculum scheduling; warm-up for physics and causal losses |
| Validation | No objective dominates training for prolonged periods; gradient norms monitored |

### 31.3 H3: Process-Model Baseline Absent — RESOLVED

| Aspect | Detail |
|---|---|
| Problem | DSSAT/APSIM baseline missing |
| Resolution | Add simple physics baselines (GDD, SPI/SPEI/VHI); DSSAT/APSIM if data allow |
| Validation | Model outperforms or complements simple physics baselines |

### 31.4 H4: Missing Modality Protocol — RESOLVED

| Aspect | Detail |
|---|---|
| Problem | No handling of missing modalities |
| Resolution | Modality masks + learned missing-modality tokens + modality dropout |
| Validation | Graceful degradation under missing modalities |

### 31.5 H5: Tier 3 Validation Undefined — RESOLVED

| Aspect | Detail |
|---|---|
| Problem | Physics and causal validation metrics were undefined |
| Resolution | Full metrics defined: water-balance residual, impossible-state rate, sign checks, monotonicity, invariance |
| Validation | Physics and causal claims testable with explicit acceptance criteria |

---

## 32. Medium-Priority Resolutions (Resolved)

| Issue | Resolution |
|---|---|
| SSL collapse risk | Embedding variance/rank/covariance spectrum monitoring with restart protocol |
| Negative transfer | Single-objective ablations; remove harmful objectives |
| Cloud contamination | Cloud threshold + SAR/monthly fallback; report valid-observation statistics |
| Temporal lag ambiguity | Lag-window ablations; select by validation performance |
| Crop-mask noise | Confidence filtering; sensitivity testing with high-confidence pixels |
| Yield scale mismatch | Aggregate predictions to label scale for validation |
| Calibration missing | ECE, Brier score, reliability curves, prediction interval coverage |
| Compute unknown | GPU-hour and storage estimates; compute budget locked before training |
| Domain knowledge gap | Agronomy/drought/phenology reading track in Weeks 9-12 |

---

## 33. Terminology Corrections

| Original Term | Corrected Term | Reason |
|---|---|---|
| Causal foundation model | Causally informed agro-ecosystem foundation prototype | Avoids overclaiming until causal validation is strong |
| Food-security risk | Agricultural production risk / yield-shortfall risk | Food security requires socioeconomic data not in model |
| Counterfactual SSL | Physics-constrained scenario augmentation | Synthetic interventions are not strict SSL |
| Global cross-region generalization | Transfer across selected contrasting agro-climatic regions | Makes claim testable within available time/data |
| Irrigation detection | Latent unobserved water-input / residual-water anomaly inference | Residuals are not uniquely irrigation |
| Counterfactual inference | Scenario simulation (Year 1) | Until validated with intervention-style evidence |

---

## 34. Reference Corrections

| Reference | Original | Correction |
|---|---|---|
| Ref [10] | Reichstein et al., 2020 | Year should be **2019** (Reichstein, M., et al. "Deep learning and process understanding for data-driven Earth system science." Nature, 2019) |
| Ref [13] | Xiong et al., arXiv:2210.00035 | Author should be verified — likely **Xia et al.** (check: Xia, J., et al. "Whu-oes: An open earth science dataset for remote sensing..." or similar) |

---

# APPENDICES

---

## Appendix A: Complete Terminology Mapping

| Concept | Final Term | Context |
|---|---|---|
| The model | AgroEarthFM | Self-Supervised Multimodal Physics-Informed Causal Geo Foundation Model |
| First-year scope | MVRC | Minimum Viable Research Contribution |
| Architecture approach | Staged implementation | Components added sequentially with gating |
| Causal framework | Temporal DAG | Time-indexed directed acyclic graph |
| Physics approach | Weak uncertainty-weighted regularization | Not hard constraints; soft penalties with uncertainty weighting |
| SSL approach | Process-centered | Objectives designed around agro-ecosystem processes |
| Data approach | Multi-resolution datacube | Variables at native scales; scale-aware fusion |
| Irrigation handling | I_latent | Latent residual water-input inference |
| Counterfactual | Scenario augmentation | Year 1 only; not validated counterfactual inference |
| Evaluation | 3-tier + calibration | Representation, downstream, physics/causal, calibration |
| Output scope | Agricultural production risk | Not food-security risk |

---

## Appendix B: Complete Variable Catalog

### Satellite Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| B2 (Blue, 490nm) | Sentinel-2 L2A | 10m | 5-day |
| B3 (Green, 560nm) | Sentinel-2 L2A | 10m | 5-day |
| B4 (Red, 665nm) | Sentinel-2 L2A | 10m | 5-day |
| B5 (Red Edge 1, 705nm) | Sentinel-2 L2A | 20m | 5-day |
| B6 (Red Edge 2, 740nm) | Sentinel-2 L2A | 20m | 5-day |
| B7 (Red Edge 3, 783nm) | Sentinel-2 L2A | 20m | 5-day |
| B8 (NIR, 842nm) | Sentinel-2 L2A | 10m | 5-day |
| B8A (Narrow NIR, 865nm) | Sentinel-2 L2A | 20m | 5-day |
| B11 (SWIR 1, 1610nm) | Sentinel-2 L2A | 20m | 5-day |
| B12 (SWIR 2, 2190nm) | Sentinel-2 L2A | 20m | 5-day |
| HLS reflectance | HLSL30/HLSS30 | 30m | 5-day |

### Derived Vegetation Indices

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| NDVI | Computed from S2 | 10m | 5-day |
| EVI | Computed from S2 | 10m | 5-day |
| NDWI | Computed from S2 | 10m | 5-day |
| LAI | PROSAIL inversion from S2 | 10m | 5-day |
| FAPAR | PROSAIL inversion from S2 | 10m | 5-day |

### Climate Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| T_min | ERA5 | ~31km | Daily → 5-day |
| T_max | ERA5 | ~31km | Daily → 5-day |
| T_mean | ERA5 | ~31km | Daily → 5-day |
| Total precipitation | ERA5/CHIRPS | ~31km/~5km | Daily → 5-day |
| Wind speed (u, v) | ERA5 | ~31km | Daily → 5-day |
| Relative humidity | ERA5 | ~31km | Daily → 5-day |
| Solar radiation | ERA5 | ~31km | Daily → 5-day |
| ET0 (Penman-Monteith) | Computed from ERA5 | ~31km | Daily → 5-day |

### Soil Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| Clay content | SoilGrids | ~250m | Static |
| Sand content | SoilGrids | ~250m | Static |
| Organic carbon | SoilGrids | ~250m | Static |
| pH | SoilGrids | ~250m | Static |
| CEC | SoilGrids | ~250m | Static |
| Bulk density | SoilGrids | ~250m | Static |
| Soil depth | SoilGrids | ~250m | Static |

### Soil Moisture Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| Surface soil moisture | SMAP L3 Enhanced | ~9km (3km enhanced) | 2-3 day |
| Surface soil moisture | ESA CCI | ~25km | Daily |

### Crop Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| Crop type | CDL/ESA WorldCereal/ISRO | 10-30m | Annual |
| Planting date | GEOGLAM crop calendar | Regional | Annual |
| Expected harvest date | GEOGLAM crop calendar | Regional | Annual |

### Label Variables

| Variable | Source | Resolution | Temporal |
|---|---|---|---|
| Yield (t/ha) | USDA NASS / ICRISAT / Eurostat | County/District/NUTS2 | Annual |
| Phenology transitions (pseudo) | NDVI/EVI derivative analysis | 10m | Per season |
| Drought/stress events | VHI/SPEI threshold | Varies | Event-based |

---

## Appendix C: Data License Summary

| Data Source | License | Access Restriction |
|---|---|---|
| Sentinel-2 | CC-BY-SA 4.0 / Copernicus | Free, open access |
| HLS | Public domain (NASA/USGS) | Free, open access |
| Landsat 8/9 | Public domain (USGS) | Free, open access |
| ERA5 | CC-BY-4.0 (Copernicus) | Free with registration |
| CHIRPS | Public domain | Free, open access |
| SoilGrids | CC-BY 4.0 (ISRIC) | Free, open access |
| SMAP | NASA Earthdata | Free with registration |
| ESA CCI SM | CC-BY-SA 4.0 | Free, open access |
| GRACE | NASA JPL | Free with registration |
| GEOGLAM | Varies | Generally open |
| USDA NASS | Public domain (US Gov) | Free, open access |
| ICRISAT | Varies | Generally open for research |
| Eurostat | CC-BY 4.0 | Free with attribution |
| CDL | Public domain (USDA) | Free, open access |
| ESA WorldCereal | ESA license | Free for research |

---

## Appendix D: Domain Knowledge Checklist (Weeks 9-12)

The researcher must complete the following reading/understanding track before Phase 3 implementation:

### D.1 Agronomy Fundamentals

| Topic | Minimum Understanding |
|---|---|
| Crop phenology stages | Wheat: Tillering, Stem Extension, Heading, Flowering, Grain Fill, Maturity; Maize: Emergence, Vegetative (V1-VT), Silking (R1), Grain Fill (R2-R6), Maturity |
| GDD / thermal time | Accumulated growing degree days; base temperatures; role in phenology prediction |
| Crop stress mechanisms | Drought, heat, combined stress; stage-specific sensitivity; stress accumulation effects |
| Yield formation | Source-sink relationships; yield components (ears/m², grains/ear, grain weight); critical periods |

### D.2 Drought Science

| Topic | Minimum Understanding |
|---|---|
| Drought types | Meteorological, agricultural, hydrological, socioeconomic |
| Drought indices | SPI, SPEI, VHI, PDSI, soil moisture percentiles; calculation and interpretation |
| Drought propagation | Climate → Soil moisture → Vegetation → Crop → Yield; time lags and memory effects |
| ET and water balance | Penman-Monteith ET; FAO crop coefficients; soil water balance components |

### D.3 Remote Sensing for Agriculture

| Topic | Minimum Understanding |
|---|---|
| Sentinel-2 for agriculture | Band selection, spectral indices, compositing, cloud masking |
| Phenology from satellite | NDVI/EVI time-series analysis; transition detection methods; noise and uncertainty |
| SAR for agriculture | Sentinel-1 backscatter; soil moisture estimation; cloud-penetrating capability |
| Data harmonization | HLS harmonization approach; cross-sensor calibration; temporal gap-filling |

### D.4 Foundation Models and SSL

| Topic | Minimum Understanding |
|---|------|
| Geospatial foundation models | SatMAE, DOFA, PhilEO, TerraFM, Presto, etc. |
| SSL methods | MAE, SimCLR, DINO, BYOL; masked reconstruction vs. contrastive learning |
| Multimodal SSL | Cross-modal prediction; modality alignment; missing modality handling |
| Transfer learning | Linear probing, fine-tuning, domain adaptation, OOD evaluation |

---

## Appendix E: Key Equations Summary

### E.1 Water Balance

```
ΔSM = P + I_latent - ET - R - D + ε
```

### E.2 Growing Degree Days

```
GDD_t = max(0, T_mean_t - T_base)
AGDD_t = Σ GDD_τ for τ from planting date to t
```

### E.3 Total Loss

```
L_total = Σ_i α_i × L_SSL_i
        + λ_water(t) × L_water
        + λ_growth(t) × L_growth
        + λ_causal(t) × L_temporal_DAG
        + λ_calib × L_calibration
```

### E.4 Physics Loss

```
L_water = w_data × ||ΔSM_pred - (P + I_latent - ET - R - D)||²
```

### E.5 Causal Loss

```
L_temporal_DAG = λ_causal(t) × [
    Σ_edges ||predicted_effect - learned_strength × cause||
    + acyclicity_penalty
    + sign_consistency_penalty
]
```

---

## Appendix F: Final Acceptable Claim (If Results Support)

> AgroEarthFM demonstrates that joint drought-phenology process-centered multimodal representation learning, augmented with weak physical regularization, improves agricultural stress and phenology/yield-risk prediction under selected transfer and extreme-event settings compared with strong statistical, temporal, and non-physics baselines.

### Claims NOT Allowed in First-Year Output Unless Separately Validated

- The system is a fully causal foundation model.
- The system predicts food-security risk directly.
- The model generalizes globally across all climate regions.
- The counterfactual outputs are decision-grade causal estimates.
- Field-scale yield accuracy is proven using only administrative yield labels.

---

*End of AgroEarthFM Complete Finalized Specification*
