# AgroEarthFM Revised Issue-Resolution and Verification & Validation Draft

**Document purpose:** Revised pre-implementation control document for making the AgroEarthFM research plan technically sound, falsifiable, and defensible against major review objections.  
**Status:** Draft v1.0  
**Prepared for:** AgroEarthFM research implementation  
**Date:** 2026-05-22

---

## 1. Executive Position

The original AgroEarthFM concept is scientifically valuable, but the implementation plan must be narrowed, staged, and validated through explicit gates. This document addresses all major issues raised during review and converts them into concrete design corrections, validation procedures, acceptance criteria, and fallback paths.

The revised first-year objective is not to build a full global causal food-security foundation model. The revised objective is to produce a technically defensible **Minimum Viable Research Contribution** focused on multimodal drought-phenology representation learning with weak physics regularization.

No research plan can guarantee “zero failures,” but this revision removes avoidable single-point failures by:

1. Bounding the scientific claims.
2. Replacing fragile architecture assumptions with staged implementation gates.
3. Using multi-resolution data fusion instead of false 10m harmonization.
4. Reformulating the causal graph as a temporal DAG.
5. Moving food-security risk outside the first model.
6. Treating irrigation and unobserved water inputs as explicit confounders.
7. Defining baselines, metrics, ablations, and failure criteria before training.
8. Adding fallback paths for every high-risk module.

---

## 2. Revised Research Scope

### 2.1 Original Over-Broad Scope

The original plan implied a comprehensive system capable of:

- Drought modeling.
- Crop phenology modeling.
- Yield-risk prediction.
- Resilience scoring.
- Counterfactual reasoning.
- Food-security risk assessment.
- Cross-climate-region generalization.
- Physics-informed causal foundation modeling.

This scope is too large for a 52-week first implementation.

### 2.2 Revised First-Year Scope

The first-year scope is:

> Develop and validate a multimodal agro-ecosystem representation model that jointly learns drought stress and crop phenology dynamics using process-centered self-supervised learning and weak water-balance regularization, evaluated on selected crop-stress, phenology, and yield-risk tasks across two or three contrasting agro-climatic regions.

### 2.3 Revised Terminology

| Original Term | Revised Term | Reason |
|---|---|---|
| Causal foundation model | Causally informed agro-ecosystem foundation prototype | Avoids overclaiming until causal validation is strong. |
| Food-security risk | Agricultural production risk / yield-shortfall risk | Food security requires socioeconomic data not included in the first model. |
| Counterfactual SSL | Physics-constrained scenario augmentation | Synthetic interventions are not strict self-supervised learning. |
| Global cross-region generalization | Transfer across selected contrasting agro-climatic regions | Makes the claim testable within available time and data. |
| Irrigation detection | Latent unobserved water-input / residual-water anomaly inference | Water-balance residuals are not uniquely irrigation. |

---

## 3. Revised Core Hypothesis

### 3.1 Final Revised Hypothesis

> A multimodal temporal representation trained on joint drought and crop-phenology process objectives, and regularized by weak water-balance constraints, will improve crop-stress detection, phenology forecasting, and yield-risk prediction under out-of-region, out-of-year, and extreme-event validation compared with single-domain, non-physics, and conventional machine-learning baselines.

### 3.2 Atomic Testable Claims

| Claim ID | Claim | Validation Requirement |
|---|---|---|
| C1 | Joint drought + phenology modeling improves representation quality. | Compare joint model against drought-only and phenology-only models. |
| C2 | Multimodal fusion improves performance and robustness. | Compare fused model against satellite-only, weather-only, and soil-only models. |
| C3 | Process-centered SSL improves transfer. | Compare process SSL against generic MAE/contrastive SSL and supervised-from-scratch baselines. |
| C4 | Weak physics regularization improves physical consistency without harming task skill. | Compare with and without water-balance/growth constraints. |
| C5 | Temporal causally informed structure improves interpretability and scenario behavior. | Validate through temporal-DAG sign checks, invariance tests, and known-event sensitivity tests. |
| C6 | The model performs better on difficult conditions. | Evaluate on out-of-region, out-of-year, and drought-event holdouts. |

---

## 4. Minimum Viable Research Contribution

### 4.1 First-Year MVRC

The first-year publishable contribution will be:

> **A multimodal, process-centered, physics-regularized representation learning framework for drought-phenology monitoring and agricultural production-risk prediction.**

### 4.2 Included in MVRC

- Multi-resolution agro-climate datacube.
- Satellite, weather, soil, and crop-calendar inputs.
- Process-centered self-supervised objectives.
- Multimodal temporal fusion.
- Crop-stress detection.
- Phenology forecasting.
- Yield-risk or yield-anomaly prediction, if labels allow.
- Weak water-balance regularization.
- Strong baselines and ablations.
- Transfer evaluation across selected regions and years.

### 4.3 Deferred to Phase 2 / Year 2

- Full neural causal discovery.
- Strong counterfactual reasoning claims.
- Food-security risk mapping.
- Global multi-biome generalization.
- Full DSSAT/APSIM integration if detailed management data are unavailable.
- Operational decision-support deployment.

---

## 5. Issue-Resolution Matrix

### 5.1 Blocker-Level Issues

| Issue | Risk | Resolution | Validation Method | Acceptance Criterion | Fallback |
|---|---|---|---|---|---|
| B1: Bidirectional causal edge creates DAG cycle | Invalid causal inference | Replace static graph with temporal DAG | Check graph acyclicity after temporal unrolling | No same-time directed cycles | Remove causal module from MVRC and keep interpretability only |
| B2: Timeline too compressed | Incomplete prototype | Define 52-week MVRC and defer high-risk modules | Milestone review every 8 weeks | At least MVRC modules completed by Week 52 | Convert causal/counterfactual to future work |
| B3: Irrigation data gap | Misattribution of drought resilience | Add latent residual-water-input variable and irrigation proxies | Compare residual anomalies with irrigation maps/canal regions where available | Residual variable improves error analysis and does not degrade transfer | Treat irrigation as uncertainty/confounder, not model output |
| B4: Food-security risk overreach | Unsupported societal claim | Reframe as agricultural production risk | Check output definitions and data inputs | No food-security claim without socioeconomic data | Add food-security layer only in later extension |

### 5.2 High-Priority Technical Issues

| Issue | Risk | Resolution | Validation Method | Acceptance Criterion | Fallback |
|---|---|---|---|---|---|
| H1: Forced 10m data harmonization | False precision | Use multi-resolution datacube and scale-aware fusion | Compare native-scale metadata and aggregation consistency | No coarse variables interpreted as true 10m signals | Aggregate all outputs to coarser validation scale |
| H2: Loss weighting unspecified | Training instability | Define weighted multi-objective loss with schedules | Monitor gradient norms and loss scale | No objective dominates training for prolonged periods | Train objectives sequentially rather than jointly |
| H3: Missing modality behavior | Deployment fragility | Add modality masks and learned missing-modality embeddings | Modality-dropout validation | < defined performance drop under missing low-priority modalities | Restrict MVP to consistently available modalities |
| H4: Tier 3 validation undefined | Physics/causal claims untestable | Define physical consistency and causal plausibility metrics | Run with/without physics and temporal-DAG modules | Physics improves consistency without task collapse | Report physics as diagnostic only |
| H5: Process-model baseline absent | Weak physics comparison | Add simple physics baselines; DSSAT/APSIM if data allow | Compare against GDD, SPI/SPEI/VHI, water-deficit models | Model outperforms or complements simple physics baselines | State process-model comparison as future work |

### 5.3 Medium-Priority Issues

| Issue | Resolution | Validation |
|---|---|---|
| SSL collapse | Monitor embedding variance, rank, covariance spectrum, nearest-neighbor diversity | Stop/restart if dimensional collapse occurs |
| Negative transfer between SSL tasks | Train single-objective and multi-objective ablations | Remove or downweight harmful objective |
| Cloud contamination | Define cloud threshold and fallback to SAR/HLS/monthly composites | Report valid-observation statistics by region/season |
| Temporal lag ambiguity | Include lag windows and causal-lag ablations | Select lag structure by validation on held-out years |
| Crop-mask noise | Use confidence filtering and crop-mask uncertainty | Sensitivity test with high-confidence crop pixels only |
| Yield-label scale mismatch | Aggregate predictions to label scale | Validate only at the scale of ground truth |
| Calibration missing | Add uncertainty and calibration metrics | Use ECE, Brier score, reliability curves |
| Compute uncertainty | Estimate GPU-hours and storage before training | Do not start full model without compute budget |
| Domain knowledge gap | Add agronomy/drought/phenology reading track | Researcher must complete domain checklist before Phase 3 |

---

## 6. Revised Data Strategy

### 6.1 Multi-Resolution Datacube Principle

The model will not force all data to 10m. Each variable will be represented at its appropriate spatial and temporal scale.

| Data Type | Native/Practical Scale | Treatment |
|---|---:|---|
| Sentinel-2 / HLS optical data | 10–30m | Fine-resolution visual/vegetation tokens |
| Landsat | 30m | Harmonized optical history where needed |
| Vegetation indices | 10–30m or aggregated | Fine temporal vegetation trajectories |
| SAR, if used | 10–30m | Cloud-robust structural/moisture proxy |
| SoilGrids | ~250m | Static soil-context tokens |
| ERA5 | coarse climate scale | Coarse temporal weather tokens |
| CHIRPS rainfall | coarse/intermediate | Rainfall forcing tokens |
| SMAP / ESA CCI soil moisture | coarse | Coarse soil-moisture context, not field truth |
| GRACE groundwater | very coarse | Optional regional context only; not field-scale input |
| Crop calendar | regional/crop-level | Prior embedding, not exact field label |
| Crop mask | field/pixel depending on source | Input with uncertainty/confidence flag |
| Yield labels | field/admin depending on source | Validation only at matching aggregation level |

### 6.2 Data Readiness Requirements

Before model training, produce a **Data Readiness Report** containing:

1. Study region boundaries.
2. Crop selection.
3. Data-source list and access method.
4. Spatial resolution of each source.
5. Temporal resolution of each source.
6. Missingness profile.
7. Cloud valid-observation statistics.
8. Crop-mask confidence.
9. Yield-label scale and availability.
10. Irrigation proxy availability.
11. Data license constraints.
12. Storage estimate.

### 6.3 Study Region Selection Criteria

A region is eligible only if it satisfies:

| Criterion | Minimum Requirement |
|---|---|
| Crop map | Available and reasonably reliable |
| Satellite observations | Sufficient cloud-free or SAR/HLS fallback |
| Weather data | Available for all study years |
| Yield or stress labels | Available at field/admin/event level |
| Drought events | At least one known dry/stress period preferred |
| Irrigation information | Map, proxy, or known irrigated/rainfed contrast preferred |
| Temporal range | Multiple seasons, ideally 5+ years |

---

## 7. Revised Architecture

### 7.1 MVP Architecture

The first implementation will use the following staged architecture:

```text
Fine Satellite Encoder
        │
Weather / Climate Temporal Encoder
        │
Soil / Static Context Encoder
        │
Crop Calendar / Crop Type Embedding
        │
Scale-Aware Fusion Module
        │
Temporal Memory Module
        │
Shared Agro-Ecosystem State Representation
        │
 ┌───────────────┬────────────────┬────────────────┐
 │ Crop Stress   │ Phenology       │ Yield Risk /   │
 │ Head          │ Forecast Head   │ Yield Anomaly  │
 └───────────────┴────────────────┴────────────────┘
        │
Weak Physics Regularization during training
```

### 7.2 Components Deferred from MVP

| Component | Status | Reason |
|---|---|---|
| Full spatial topology GNN | Optional extension | Requires robust graph construction from DEM/drainage/field adjacency |
| Full neural causal discovery | Deferred | High validation burden |
| Counterfactual reasoning module | Deferred or limited demo | Requires validated temporal-DAG and physics layer |
| Food-security output head | Removed from MVP | Requires socioeconomic data |

### 7.3 Soil GNN and Spatial Topology Resolution

The original plan had both a soil GNN and a spatial-topology GNN. To avoid redundancy:

- MVP will use a single **spatial-context encoder** if graph features are used.
- Soil properties become node features.
- Topology, adjacency, slope, hydrology, or distance relationships become edge features.
- If graph construction is unreliable, use patch-level static soil/topography embeddings instead.

### 7.4 Missing Modality Protocol

The model must support missing or unavailable inputs.

For each modality:

- Add modality-present mask.
- Add learned missing-modality token.
- Use modality dropout during training.
- Report performance under missing-modality scenarios.

Validation:

| Test | Description |
|---|---|
| Full modality test | All inputs available |
| No soil moisture test | Remove SMAP/ESA CCI |
| No irrigation proxy test | Remove irrigation-related features |
| No SAR test, if SAR included | Test optical-only fallback |
| Cloud-degraded optical test | Simulate missing optical observations |

---

## 8. Revised Physics Framework

### 8.1 Water-Balance Constraint

The revised water-balance relationship is:

```text
ΔSM = P + I_latent - ET - R - D + ε
```

Where:

| Symbol | Meaning |
|---|---|
| ΔSM | Soil moisture change |
| P | Precipitation |
| I_latent | Latent unobserved water input, including possible irrigation or groundwater contribution |
| ET | Evapotranspiration |
| R | Runoff |
| D | Drainage/deep percolation |
| ε | Measurement/model residual |

### 8.2 Key Correction

The residual term must not be interpreted automatically as irrigation. It is a latent anomaly that may include irrigation, groundwater, model error, or measurement error.

### 8.3 Physics Loss Design

The physics loss will be uncertainty-weighted:

```text
L_water = w_data × ||ΔSM_pred - (P + I_latent - ET - R - D)||
```

Where `w_data` is lower when data uncertainty is high.

### 8.4 Plant Growth Constraint

The plant-growth constraint will include:

- Growing Degree Days.
- Water availability.
- Solar radiation.
- Phenology-stage-specific stress sensitivity.

GDD must be explicitly accumulated:

```text
GDD_t = max(0, T_mean_t - T_base)
AGDD_t = Σ GDD_τ for τ from planting date to t
```

### 8.5 Physics Validation Metrics

| Metric | Purpose |
|---|---|
| Water-balance residual RMSE | Measures physical consistency |
| Residual bias by region/season | Detects systematic physics misspecification |
| Impossible-state rate | Counts predictions violating basic physical logic |
| Stress-under-deficit check | Tests whether healthy growth is predicted under severe deficit without residual explanation |
| Physics-task tradeoff | Checks whether physics improves consistency without destroying accuracy |

### 8.6 Physics Acceptance Criteria

The physics-regularized model is accepted only if:

1. Water-balance residual improves relative to non-physics model.
2. Crop-stress/phenology/yield-task performance does not degrade beyond a predefined tolerance.
3. Out-of-region or extreme-event performance improves or remains stable.
4. Residual anomalies are interpretable and not random noise.

If these conditions fail, physics will be reported as diagnostic rather than used as a core claim.

---

## 9. Revised Causal Framework

### 9.1 Causal Claim Reframing

The first version will not claim full causal discovery or fully validated counterfactual inference. It will claim:

> AgroEarthFM uses a causally informed temporal structure to guide representation learning and scenario analysis.

### 9.2 Temporal DAG

The revised causal structure is time-indexed:

```text
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

A cleaner structural summary:

```text
Weather_t → ET_t
Rainfall_t, ET_t, Soil_t, I_latent_t → Soil_Moisture_t
Soil_Moisture_t, Temperature_t, Crop_Stage_t → Vegetation_Health_t
Vegetation_Health_t, AGDD_t, Crop_Calendar → Crop_Stage_t+1
Vegetation_Health_t, Crop_Stage_t, Stress_Accumulation_t → Yield_Risk_t+1
```

### 9.3 Causal Validation Metrics

The causal component will not be judged only by edge-recovery F1. Instead, use:

| Metric/Test | Purpose |
|---|---|
| Graph acyclicity after temporal unrolling | Ensures structural validity |
| Directional sign checks | Rainfall should not reduce soil moisture under normal conditions, etc. |
| Invariance across years | Tests stable relationships |
| Invariance across regions | Tests transfer of causal relationships |
| Extreme event behavior | Tests model under known drought years |
| Irrigated vs rainfed contrast | Tests confounding handling |
| Scenario monotonicity | Less rainfall should generally not produce lower stress unless latent input compensates |
| Known-event case studies | Compare model explanations against documented drought/heat events |

### 9.4 Counterfactual Module Status

Counterfactual reasoning will be handled in stages:

| Stage | Status |
|---|---|
| Year 1 | Limited scenario-sensitivity tests only |
| Year 2 | Formal counterfactual module if validation data and causal assumptions are adequate |

Counterfactual outputs must be labeled as **scenario simulations** unless validated using intervention-like evidence.

---

## 10. Revised Self-Supervised Learning Plan

### 10.1 SSL Objective Sequence

Do not train all objectives at once initially. Introduce them progressively.

| Stage | Objective | Purpose |
|---|---|---|
| SSL-1 | Masked spatiotemporal reconstruction | Learn basic EO representations |
| SSL-2 | Cross-modal masked prediction | Learn relationships among weather, soil, and vegetation |
| SSL-3 | Future vegetation trajectory prediction | Learn temporal ecosystem dynamics |
| SSL-4 | Phenology pseudo-transition prediction | Learn crop-stage timing |
| SSL-5 | Drought stress trajectory prediction | Learn drought response dynamics |
| Scenario-1 | Physics-constrained scenario augmentation | Later-stage scenario consistency, not strict SSL |

### 10.2 Loss Function

Final training loss:

```text
L_total = Σ_i α_i L_SSL_i
        + λ_water(t) L_water
        + λ_growth(t) L_growth
        + λ_causal(t) L_temporal_DAG
        + λ_calib L_calibration
```

### 10.3 Training Schedule

| Phase | Active Losses |
|---|---|
| Warm-up | SSL reconstruction / future prediction only |
| Fusion phase | Add cross-modal prediction |
| Process phase | Add phenology and drought trajectory losses |
| Physics phase | Add water/growth losses with low λ and warm-up |
| Causal-informed phase | Add temporal-DAG consistency only if previous phases pass |

### 10.4 SSL Stability Monitoring

Monitor:

- Embedding variance.
- Embedding rank.
- Collapse metrics.
- Objective-specific validation losses.
- Gradient norms.
- Gradient cosine similarity between objectives.
- Linear-probe performance over time.

Acceptance:

- No dimensional collapse.
- Representation improves over supervised-from-scratch baseline.
- No single SSL objective degrades all downstream tasks.

---

## 11. Evaluation and Validation Plan

### 11.1 Required Baselines

| Baseline | Purpose |
|---|---|
| Climatology | Minimum skill baseline |
| Persistence model | Temporal forecasting baseline |
| GDD phenology model | Simple crop-physics baseline |
| SPI/SPEI/VHI drought index models | Drought-monitoring baselines |
| Random Forest / XGBoost | Strong tabular engineered-feature baseline |
| LSTM / TCN | Deep temporal baseline |
| Satellite-only model | Tests value of imagery alone |
| Weather-only model | Tests value of climate data alone |
| Multimodal no-physics model | Tests physics contribution |
| Multimodal no-process-SSL model | Tests process SSL contribution |
| Existing GeoFM features, if feasible | Tests value over general EO representations |
| AquaCrop/DSSAT/APSIM, if feasible | Optional process-model comparison if data allow |

### 11.2 Required Splits

| Split | Purpose |
|---|---|
| Random within-region split | Basic sanity, not final proof |
| Out-of-year split | Temporal generalization |
| Out-of-region split | Spatial transfer |
| Extreme-event holdout | Drought/heat robustness |
| Early-season prediction split | No-leakage forecasting |
| Irrigated/rainfed contrast, if available | Confounding analysis |

### 11.3 Task Metrics

| Task | Metrics |
|---|---|
| Crop stress detection | AUROC, AUPRC, F1, event-level recall, spatial IoU |
| Phenology forecasting | MAE in days, transition F1, stage accuracy, calibration |
| Drought forecasting | RMSE for drought index, onset F1, lead-time skill score |
| Yield-risk prediction | RMSE/MAE/R² for yield, AUC/F1 for below-threshold risk |
| Representation quality | Linear probing, fine-tuning efficiency, transfer gap |
| Physics consistency | Water-balance residual, impossible-state rate |
| Causal plausibility | Sign checks, monotonicity, invariance, known-event agreement |
| Calibration | Brier score, ECE, reliability curves, prediction interval coverage |

### 11.4 No-Leakage Protocol

For forecasting tasks, the following are prohibited:

- Future satellite observations beyond forecast date.
- Full-season composites in early-season predictions.
- Yield labels or statistics from the target prediction period.
- Gap-filled products that use future observations unless explicitly allowed and documented.
- Region-level normalization using held-out region statistics.

All preprocessing must be fit only on training data and applied to validation/test data.

---

## 12. Acceptance Gates

### Gate G0 — Scope Lock

**Must produce:**

- Final crop selection.
- Final region selection.
- Final target years.
- Final task list.
- Final data list.
- Final metrics.

**Go condition:** Scope is testable and data exists.

---

### Gate G1 — Data Readiness

**Go condition:**

- At least two regions have sufficient data.
- Cloud/missingness profile is acceptable or fallback exists.
- Yield/phenology/stress labels are available or valid proxy labels are defined.
- Scale alignment is documented.

**No-go condition:** No reliable labels or severe missingness with no fallback.

---

### Gate G2 — Baseline Performance

**Go condition:**

- At least one simple baseline beats climatology/persistence.
- Evaluation pipeline is stable.

**No-go condition:** Labels or task definition appear invalid.

---

### Gate G3 — Multimodal Fusion Value

**Go condition:**

- Fusion improves over best single-modality baseline in at least one primary task or transfer split.

**No-go condition:** Fusion adds no value after fixing alignment and missingness.

---

### Gate G4 — Process SSL Value

**Go condition:**

- Process SSL improves transfer, fine-tuning efficiency, or extreme-event performance over generic SSL or supervised training.

**No-go condition:** SSL collapses or provides no measurable benefit.

---

### Gate G5 — Physics Value

**Go condition:**

- Physics improves consistency and does not materially damage predictive performance.

**No-go condition:** Physics worsens both physical residuals and task metrics.

---

### Gate G6 — Causal-Informed Value

**Go condition:**

- Temporal-DAG module passes sign, monotonicity, and invariance checks.

**No-go condition:** Scenario behavior is unstable or contradicts domain knowledge.

---

### Gate G7 — Research Readiness

**Go condition:**

- Ablations support at least one clear contribution.
- Results are reproducible.
- Failure modes are documented.
- Claims are bounded by evidence.

---

## 13. Revised 52-Week Roadmap

| Weeks | Phase | Output |
|---:|---|---|
| 1–2 | Revision and scope lock | Revised hypothesis, MVRC, study regions, tasks, metrics |
| 3–8 | Data audit and datacube prototype | Multi-resolution datacube for one pilot region |
| 9–12 | Domain and tool strengthening | Agronomy/drought/phenology knowledge track completed |
| 13–18 | Baselines | Climatology, persistence, XGBoost/RF, GDD/SPI/VHI, LSTM/TCN |
| 19–26 | MVP model | Satellite/weather/soil/crop encoders and fusion module |
| 27–34 | Process-centered SSL | Future trajectory, phenology, drought objectives |
| 35–40 | Physics regularization | Water-balance and growth-loss ablations |
| 41–45 | Transfer and robustness testing | Out-of-region/year/event evaluation |
| 46–49 | Causal-informed diagnostics | Temporal-DAG sign/invariance/scenario tests, if earlier gates pass |
| 50–52 | Final integration and report | Ablation tables, error analysis, manuscript-ready results |

---

## 14. Reproducibility and Compute Plan

Before full training, document:

1. Hardware used.
2. GPU memory requirement.
3. Estimated GPU-hours per experiment.
4. Dataset size on disk.
5. Preprocessing time.
6. Random seeds.
7. Software versions.
8. Data download scripts.
9. Train/validation/test split files.
10. Model configuration files.
11. Experiment tracking system.

Minimum reproducibility requirement:

- Every final table must be reproducible from committed configuration files and fixed data splits.

---

## 15. Ethical and Communication Constraints

The first-year model must not be presented as an operational food-security decision system.

Allowed language:

- Agricultural production-risk prototype.
- Crop-stress monitoring model.
- Yield-shortfall risk estimator.
- Causally informed scenario analysis.

Avoid language:

- Food-security crisis prediction.
- Fully causal decision engine.
- Guaranteed climate adaptation recommendation.
- Field-level yield truth if labels are administrative.

All maps and risk outputs should include uncertainty and caveats.

---

## 16. Final Revised Implementation Decision

The revised plan is technically sound enough to proceed if and only if the following pre-implementation artifacts are completed:

1. Scope-lock document.
2. Data Readiness Report.
3. Baseline Benchmark Plan.
4. Multi-resolution datacube specification.
5. Temporal-DAG specification.
6. Loss-schedule specification.
7. Physics-validation protocol.
8. Causal-diagnostic protocol.
9. No-leakage evaluation protocol.
10. Compute and reproducibility plan.

Once these are complete, Phase 1 implementation can begin with significantly reduced risk of single-point failure.

---

## 17. Final Claim Boundary

The first implementation should claim only what it validates.

### Acceptable Final Claim if Results Support It

> AgroEarthFM demonstrates that joint drought-phenology process-centered multimodal representation learning, augmented with weak physical regularization, improves agricultural stress and phenology/yield-risk prediction under selected transfer and extreme-event settings compared with strong statistical, temporal, and non-physics baselines.

### Claims Not Allowed in First-Year Output Unless Separately Validated

- The system is a fully causal foundation model.
- The system predicts food-security risk directly.
- The model generalizes globally across all climate regions.
- The counterfactual outputs are decision-grade causal estimates.
- Field-scale yield accuracy is proven using only administrative yield labels.

---

## 18. Summary of Resolved Issues

| Original Weakness | Revised Resolution |
|---|---|
| Scope too broad | MVRC defined for 52 weeks |
| DAG cycle | Temporal DAG adopted |
| Food-security overreach | Reframed as agricultural production risk |
| Forced 10m alignment | Multi-resolution datacube adopted |
| Irrigation missing | Latent residual-water-input strategy added |
| Physics too rigid | Weak uncertainty-weighted physics loss added |
| Counterfactual SSL misclassified | Reframed as scenario augmentation and deferred |
| Loss function vague | Weighted scheduled loss defined |
| Tier 3 undefined | Physics and causal plausibility metrics defined |
| Timeline unrealistic | Revised staged 52-week roadmap added |
| Missing modality risk | Modality masking/dropout protocol added |
| SSL collapse risk | Representation monitoring added |
| Yield scale mismatch | Aggregation-to-label-scale rule added |
| Temporal leakage risk | No-leakage protocol added |
| Compute unknown | Reproducibility and compute plan required |

---

## 19. Closing Position

With these revisions, AgroEarthFM becomes a technically grounded and defensible research program. The project remains ambitious, but its first implementation is now bounded, testable, and protected against the most serious architectural, data, physics, causal, and evaluation failures.

The key principle going forward is:

> Build the simplest version that can prove the central scientific contribution, validate every added complexity through ablation, and never claim more than the evidence supports.
