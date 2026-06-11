# AgroEarthFM Finalised Model Architecture Specification

**Architecture version:** Year-1 MVP, technically defensible research prototype  
**Diagram file:** `AgroEarthFM_Final_Architecture_Diagram.svg`  
**Core design stance:** Multimodal drought-phenology representation learning with weak physics regularisation and causal-informed diagnostics.  
**Explicitly excluded from Year-1 MVP:** direct food-security prediction, full neural causal discovery, and decision-grade counterfactual claims unless separately validated.

---

## 1. Final Architecture Summary

The final AgroEarthFM architecture is organised into five layers:

```text
Multi-Resolution Inputs
        ↓
Quality-Controlled Multi-Resolution Datacube
        ↓
Modality-Specific Encoders
        ↓
Scale-Aware Fusion + Hierarchical Temporal Memory
        ↓
Shared Agro-Ecosystem State Representation
        ↓
Task Heads + Physics/Causal Validation
```

The architecture is designed to prevent the major failure modes identified during review:

- No false 10m precision for coarse climate/soil variables.
- No invalid same-time cyclic causal graph.
- No unsupported direct food-security output.
- No unhandled missing modality failures.
- No unvalidated counterfactual claims.
- No physics loss treated as guaranteed truth.

---

## 2. Layer 1 — Multi-Resolution Inputs

### 2.1 Fine Earth Observation / Vegetation Inputs

**Sources:**

- HLS / Sentinel-2 / Landsat.
- Optional Sentinel-1 SAR for cloudy regions.
- Derived vegetation indices: NDVI, EVI, NDWI, LAI/FAPAR where available.

**Scale:** 10–30m, retained as fine-resolution tokens.

**Purpose:**

- Vegetation condition.
- Canopy water stress.
- Crop growth trajectory.
- Phenology transitions.

---

### 2.2 Weather and Climate Forcing Inputs

**Sources:**

- ERA5, CHIRPS, or equivalent weather products.

**Variables:**

- Rainfall.
- Temperature.
- Humidity.
- Solar radiation.
- Wind speed.
- Vapor pressure deficit proxies.

**Scale:** Native/coarse scale, not forcibly upsampled to 10m.

**Purpose:**

- Climate forcing.
- Drought precursor signals.
- Heat stress.
- Evapotranspiration estimation.

---

### 2.3 Soil and Hydro-Context Inputs

**Sources:**

- SoilGrids.
- SMAP / ESA CCI soil moisture.
- DEM/topography if available.
- Optional groundwater/regional water-storage data only as coarse context.

**Purpose:**

- Soil water-holding capacity.
- Static soil context.
- Coarse soil moisture condition.
- Hydrological susceptibility.

---

### 2.4 Crop and Management Context Inputs

**Sources:**

- Crop calendars.
- Crop masks.
- Crop type maps.
- Planting windows.
- Irrigation maps/proxies where available.

**Important correction:** Irrigation is not assumed to be known. If unavailable, the model uses a latent residual-water-input variable rather than pretending irrigation is observed.

---

### 2.5 Labels and Validation Data

Possible validation labels:

- Phenology events or pseudo-events.
- Drought event records.
- Crop stress labels/proxies.
- Yield or yield anomaly labels.
- Administrative production statistics.

**Rule:** Predictions must be validated at the same scale as the label. Field-level claims cannot be made from district-level yield labels.

---

## 3. Layer 2 — Quality-Controlled Multi-Resolution Datacube

The datacube is not a single 10m tensor. It is a multi-resolution spatiotemporal structure with explicit scale metadata.

### 3.1 Datacube Responsibilities

- Preserve native resolution where scientifically necessary.
- Retain CRS and geospatial metadata.
- Align time windows without future leakage.
- Store modality-present masks.
- Store cloud and gap masks.
- Store crop-mask confidence.
- Store uncertainty information where available.
- Maintain locked train/validation/test splits.

### 3.2 Temporal Structure

| Time Path | Purpose |
|---|---|
| Fine path, 5–10 day windows | Phenology changes, short drought response, rapid stress events |
| Coarse path, monthly/seasonal windows | Climate context, cumulative drought, growing-season state |

### 3.3 Anti-Failure Rule

> Coarse variables may inform fine predictions, but they must not be interpreted as true fine-resolution observations.

---

## 4. Layer 3 — Modality-Specific Encoders

### 4.1 EO Temporal Patch Encoder

**Suggested architecture:** compact ViT, CNN-ViT hybrid, or temporal patch transformer.

**Input:** Fine EO image patches and vegetation-index trajectories.

**Output:** Vegetation/appearance/phenology tokens.

---

### 4.2 Weather Temporal Encoder

**Suggested architecture:** Temporal Transformer, TCN, or GRU/Transformer hybrid.

**Input:** Weather sequences at native/coarse resolution.

**Output:** Climate forcing tokens and lag-aware stress signals.

---

### 4.3 Soil / Static Context Encoder

**Suggested architecture:** MLP for static variables; optional shared spatial GNN if robust graph features exist.

**Input:** Soil type, water-holding capacity, static hydrological features, coarse soil moisture.

**Output:** Soil and hydro-context tokens.

---

### 4.4 Crop Calendar Encoder

**Suggested architecture:** embedding layers plus temporal calendar features.

**Input:** crop type, planting window, expected growth-stage calendar, crop mask confidence.

**Output:** Phenology-prior tokens.

### Required Feature

Accumulated Growing Degree Days must be explicitly computed or represented:

```text
GDD_t = max(0, T_mean_t - T_base)
AGDD_t = sum(GDD from planting to time t)
```

---

### 4.5 Missing-Modality Handler

Each modality has:

- Modality-present mask.
- Learned missing-modality token.
- Modality dropout during training.

This prevents the model from failing when irrigation, groundwater, SAR, or soil moisture data are unavailable.

---

## 5. Layer 4 — Scale-Aware Fusion and Temporal Memory

### 5.1 Scale-Aware Fusion Module

The fusion module uses cross-resolution attention or gated fusion.

Recommended information order:

```text
weather forcing → soil/water state → vegetation response → phenology state → yield-risk state
```

This is not treated as a full causal proof. It is an inductive bias consistent with the physical process chain.

### 5.2 Token Budget Control

The architecture must explicitly control:

- Number of EO spatial tokens.
- Number of temporal tokens.
- Number of weather/soil context tokens.
- Cross-attention pairings.
- Memory footprint.

This avoids uncontrolled transformer complexity.

---

### 5.3 Hierarchical Temporal Memory

The temporal memory module has two paths:

| Path | Role |
|---|---|
| Fine temporal path | Captures 5–10 day vegetation and stress dynamics |
| Coarse temporal path | Captures monthly and seasonal climate context |

It must also represent:

- Lagged drought effects.
- Stress accumulation.
- Recovery dynamics.
- AGDD accumulation.

---

## 6. Layer 5 — Shared Agro-Ecosystem State

The model produces a shared latent state:

```text
z_t = shared agro-ecosystem state at time t
```

This state should encode:

- Vegetation health.
- Soil-water context.
- Phenology state.
- Heat/water stress accumulation.
- Latent residual-water anomaly.
- Uncertainty.

This is the main representation used by downstream task heads.

---

## 7. Output Heads

### 7.1 Crop Stress Head

Outputs:

- Crop stress score.
- Water/heat stress probability.
- Uncertainty.

Validation:

- AUROC.
- AUPRC.
- F1.
- Event-level recall.
- Spatial IoU where applicable.

---

### 7.2 Phenology Forecast Head

Outputs:

- Current growth stage.
- Transition probability.
- Expected timing of next transition.

Validation:

- MAE in days.
- Transition F1.
- Stage accuracy.
- Calibration.

---

### 7.3 Drought Trajectory Head

Outputs:

- Stress trajectory.
- Drought onset/recovery likelihood.
- Lead-time forecast.

Validation:

- RMSE against drought/stress index.
- Onset F1.
- Lead-time skill score.

---

### 7.4 Yield-Risk / Yield-Anomaly Head

Outputs:

- Yield anomaly.
- Below-threshold yield-risk probability.
- Agricultural production-risk score.

Important:

- This is not a food-security risk head.
- Validation must occur at the same spatial scale as yield labels.

Validation:

- RMSE.
- MAE.
- R².
- AUC/F1 for below-threshold risk.
- Calibration.

---

### 7.5 Residual Water / Uncertainty Head

Outputs:

- Latent unobserved water-input anomaly.
- Predictive uncertainty.

Interpretation:

- May indicate irrigation, groundwater, data error, runoff/drainage mismatch, or unobserved management.
- Must not be automatically labeled irrigation without external validation.

---

## 8. Training Objectives

### 8.1 Sequential Process-Centered SSL

Training objectives are introduced progressively:

1. Masked spatiotemporal reconstruction.
2. Cross-modal masked prediction.
3. Future vegetation trajectory prediction.
4. Phenology pseudo-transition prediction.
5. Drought stress trajectory prediction.
6. Physics-constrained scenario augmentation only after physics validation.

---

### 8.2 Final Loss Function

```text
L_total = Σ_i α_i L_SSL_i
        + λ_water(t) L_water
        + λ_growth(t) L_growth
        + λ_DAG(t) L_temporal_DAG
        + λ_calib L_calibration
```

Where:

- `α_i` are objective-specific SSL weights.
- `λ_water(t)` is a warm-up schedule for water-balance loss.
- `λ_growth(t)` is a warm-up schedule for plant-growth constraints.
- `λ_DAG(t)` is active only after base model stability.
- `λ_calib` supports probability calibration.

---

## 9. Physics Regularisation

### 9.1 Water Balance

```text
ΔSM = P + I_latent - ET - R - D + ε
```

Where:

- `ΔSM`: soil moisture change.
- `P`: precipitation.
- `I_latent`: latent unobserved water input.
- `ET`: evapotranspiration.
- `R`: runoff.
- `D`: drainage/deep percolation.
- `ε`: residual/model uncertainty.

### 9.2 Growth Constraint

Plant growth is regularised using:

- AGDD.
- Water availability.
- Solar radiation.
- Crop-stage-specific stress sensitivity.

### 9.3 Acceptance Rule

Physics is accepted as a core contribution only if it:

1. Reduces physical inconsistency.
2. Does not materially damage task performance.
3. Improves or stabilises transfer/extreme-event performance.

---

## 10. Causal-Informed Diagnostics

The Year-1 architecture uses a temporal DAG only as a diagnostic and inductive-bias layer.

### 10.1 Temporal DAG Principle

No same-time bidirectional causal cycles are allowed.

Example:

```text
Crop_Stage_t → Vegetation_Health_t
Vegetation_Health_t → Crop_Stage_t+1
Vegetation_Health_t, Crop_Stage_t, Stress_t → Yield_Risk_t+1
```

### 10.2 Validation

Use:

- Graph acyclicity check.
- Sign checks.
- Monotonicity tests.
- Invariance across regions and years.
- Extreme-event case studies.
- Irrigated/rainfed contrast if available.

Counterfactual outputs should be called scenario simulations unless validated with intervention-like evidence.

---

## 11. Baselines and Ablations

Required baselines:

- Climatology.
- Persistence.
- GDD phenology model.
- SPI/SPEI/VHI drought models.
- Random Forest / XGBoost.
- LSTM / TCN.
- Satellite-only model.
- Weather-only model.
- Multimodal no-physics model.
- Multimodal no-process-SSL model.
- Existing EO foundation model features if feasible.
- AquaCrop/DSSAT/APSIM only if detailed process-model inputs are available.

Required ablations:

- No physics.
- No crop-calendar input.
- No weather input.
- No soil context.
- No process SSL.
- No missing-modality training.
- No temporal memory.

---

## 12. Final Architecture Decision

The final model architecture is approved for implementation as a Year-1 MVP under the following boundaries:

1. It is a multimodal, physics-regularised, causally informed research prototype.
2. It does not claim direct food-security prediction.
3. It does not claim fully validated causal counterfactual inference.
4. It validates predictions at the appropriate spatial scale.
5. It uses strong baselines and ablations before making any architectural claims.
6. It keeps failure fallbacks for physics, causal diagnostics, missing data, and label limitations.

---

## 13. One-Line Architecture Definition

> AgroEarthFM-MVP is a multi-resolution, multimodal temporal representation model that fuses satellite, weather, soil, and crop-calendar information into a shared agro-ecosystem state, trained with process-centered self-supervision and weak physics constraints to predict crop stress, phenology, drought trajectory, and agricultural production risk under rigorous transfer and extreme-event validation.
