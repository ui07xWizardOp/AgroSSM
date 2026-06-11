# AgroEarthFM Two-Page Architecture Package

**Primary visual file:** `AgroEarthFM_Two_Page_Architecture_Package.html`  
**Purpose:** Provide a clear two-page architecture package that preserves both high-level readability and detailed technical validation information.

---

## Page 1 — Clean High-Level Model Architecture

Page 1 contains the main model pipeline:

```text
Multi-Resolution Inputs
        ↓
Quality-Controlled Multi-Resolution Datacube
        ↓
Modality-Specific Encoders
        ↓
Scale-Aware Fusion + Hierarchical Temporal Memory
        ↓
Shared Agro-Ecosystem State z_t
        ↓
Output Heads
```

### Information included on Page 1

1. **Multi-resolution inputs**
   - EO / vegetation: HLS, Sentinel-2, Landsat, optional SAR, NDVI/EVI/NDWI.
   - Weather / climate: rainfall, temperature, humidity, radiation, wind, VPD/ET drivers.
   - Soil / hydro: SoilGrids, SMAP/ESA CCI, soil texture, water capacity, optional topography.
   - Crop context: crop type, crop calendar, planting window, crop-mask confidence, irrigation proxy if available.
   - Validation labels: phenology, stress, yield, drought-event records.

2. **Quality-controlled multi-resolution datacube**
   - No forced 10m upscaling.
   - Native scale metadata retained.
   - Cloud/gap/crop masks.
   - Modality-present masks.
   - No-leakage train/validation/test splits.

3. **Encoders**
   - EO temporal encoder.
   - Weather temporal encoder.
   - Soil/static encoder.
   - Crop-calendar encoder.
   - Missing-modality handler.

4. **Fusion and memory**
   - Scale-aware ordered fusion.
   - Hierarchical temporal memory.
   - Process-informed order: weather → soil/water → vegetation → phenology.
   - Fine 5–10 day and coarse seasonal temporal paths.

5. **Shared state**
   - Vegetation health.
   - Soil-water state.
   - Phenology state.
   - Stress accumulation.
   - Latent residual-water anomaly.
   - Predictive uncertainty.

6. **Output heads**
   - Crop stress.
   - Phenology forecast.
   - Drought trajectory.
   - Yield-risk / anomaly.
   - Residual water + uncertainty.

---

## Page 2 — Detailed Training, Physics, Causal Diagnostics, and Validation Controls

Page 2 contains the detailed technical safeguards and validation logic.

### Information included on Page 2

1. **Training objectives**
   - Masked spatiotemporal reconstruction.
   - Cross-modal masked prediction.
   - Future vegetation trajectory prediction.
   - Phenology pseudo-transition prediction.
   - Drought stress trajectory prediction.
   - Scenario augmentation only after physics validation.

2. **Loss design**

```text
L_total = Σ α_i L_SSL_i
        + λ_water(t) L_water
        + λ_growth(t) L_growth
        + λ_DAG(t) L_DAG
        + λ_calib L_calib
```

3. **Physics constraints**

```text
ΔSM = P + I_latent − ET − R − D + ε
```

Where `I_latent` is an unobserved water-input term that may represent irrigation, groundwater, residual mismatch, or data uncertainty. It must not be automatically interpreted as irrigation.

4. **Plant-growth constraint**
   - AGDD.
   - Water availability.
   - Radiation.
   - Crop-stage-specific stress sensitivity.

5. **Temporal-DAG correction**

```text
CropStage_t → VegetationHealth_t
VegetationHealth_t → CropStage_t+1
VegetationHealth_t, CropStage_t, Stress_t → YieldRisk_t+1
```

6. **Validation matrix**
   - Tests each major research claim.
   - Identifies required baselines.
   - Defines acceptance signals.

7. **Task metrics**
   - Crop stress: AUROC, AUPRC, F1, event recall, spatial IoU.
   - Phenology: MAE days, transition F1, stage accuracy, calibration.
   - Drought: RMSE, onset F1, lead-time skill.
   - Yield-risk: MAE/RMSE/R², AUC/F1, calibration.
   - Physics: water-balance residual, impossible-state rate.
   - Calibration: Brier score, ECE, reliability curves.

8. **Baselines and ablations**
   - Climatology.
   - Persistence.
   - GDD model.
   - SPI/SPEI/VHI drought models.
   - RF/XGBoost.
   - LSTM/TCN.
   - Single-modality baselines.
   - No-physics ablation.
   - No-process-SSL ablation.
   - No crop-calendar/weather/soil ablations.

9. **Acceptance gates**
   - G0: scope lock.
   - G1: data readiness.
   - G2: baseline validity.
   - G3: multimodal fusion value.
   - G4: process SSL value.
   - G5: physics value.
   - G6: temporal-DAG diagnostic value.
   - G7: research-readiness and reproducibility.

10. **Anti-failure rules**
   - No false precision.
   - No temporal leakage.
   - No food-security overclaiming.
   - No unsupported causal counterfactual claims.

---

## Claim Boundary Preserved

The two-page package preserves the technically sound claim boundary:

> AgroEarthFM-MVP is a multi-resolution, multimodal temporal representation model that fuses satellite, weather, soil, and crop-calendar information into a shared agro-ecosystem state, trained with process-centered self-supervision and weak physics constraints to predict crop stress, phenology, drought trajectory, and agricultural production risk under rigorous transfer and extreme-event validation.

The package does **not** claim:

- Direct food-security prediction.
- Fully validated causal counterfactual inference.
- Global generalization across all climate regions.
- Field-level yield truth from administrative yield labels.
- Automatic irrigation detection from residual water anomalies.
