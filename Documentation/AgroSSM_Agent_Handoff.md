# AgroSSM Agent Handoff: Step-by-Step Implementation Instructions

**Purpose:** This document is a self-contained, exhaustive instruction set for any AI agent to implement the AgroSSM research prototype from scratch. It assumes the agent has no prior context about the project. Every file, function, tensor shape, test, and decision point is specified explicitly so that implementation proceeds without ambiguity.

**How to use this document:**
1. Read Section 1 (Context) to understand what you're building.
2. Read Section 2 (Invariants) â€” these are rules you must NEVER violate.
3. Execute Sections 3â€“10 in order. Each section is one implementation phase.
4. Within each section, implement files in the listed order â€” dependencies go first.
5. After writing each file, run the specified test command. Do not proceed until tests pass.

---

## TABLE OF CONTENTS

- [1. Context: What You Are Building](#1-context-what-you-are-building)
- [2. Invariants: Rules That Must Never Be Violated](#2-invariants-rules-that-must-never-be-violated)
- [3. Phase 0: Project Scaffolding](#3-phase-0-project-scaffolding)
- [4. Phase 1A: Configuration Files](#4-phase-1a-configuration-files)
- [5. Phase 1B: Data Pipeline â€” Datacube and Patch Extraction](#5-phase-1b-data-pipeline--datacube-and-patch-extraction)
- [6. Phase 2A: Unified Wavelength-Conditioned Tokenizer](#6-phase-2a-unified-wavelength-conditioned-tokenizer)
- [7. Phase 2B: Hybrid Temporal Backbone (SSM + SWA + MLA)](#7-phase-2b-hybrid-temporal-backbone-ssm--swa--mla)
- [8. Phase 2C: Regime-Routed Mixture-of-Experts](#8-phase-2c-regime-routed-mixture-of-experts)
- [9. Phase 2D: Output Heads and Physics-as-State-Regularizer](#9-phase-2d-output-heads-and-physics-as-state-regularizer)
- [10. Phase 2E: Full AgroSSM Model Assembly](#10-phase-2e-full-agrossm-model-assembly)
- [11. Phase 3: SSL Pretext Tasks](#11-phase-3-ssl-pretext-tasks)
- [12. Phase 4: Training Pipeline](#12-phase-4-training-pipeline)
- [13. Phase 5: Baselines](#13-phase-5-baselines)
- [14. Phase 6: Evaluation and Diagnostics](#14-phase-6-evaluation-and-diagnostics)
- [15. Failure Recovery Playbook](#15-failure-recovery-playbook)
- [16. File Index â€” Every File You Will Create](#16-file-index--every-file-you-will-create)

---

## 1. Context: What You Are Building

You are building **AgroSSM** â€” a PyTorch model that:
- Reads multi-resolution Earth observation data (satellite images at 10m, weather at 31km, soil at 250m, crop calendars).
- Processes them through a **unified tokenizer** (one encoder for all modalities).
- Runs a **hybrid temporal backbone** that interleaves State-Space Model blocks (linear-time physical accumulators), Sliding-Window Attention blocks (local spatial patterns), and Multi-Head Latent Attention blocks (coarse-context compression).
- Routes each token through a **Mixture-of-Experts** (8 experts, top-1 routing) that specializes by agro-climatic regime.
- Produces predictions from 5 output heads: Crop Stress, Phenology Forecast, Yield-Risk (primary), and Drought Trajectory, Residual Water/Uncertainty (auxiliary).
- Is trained with **physics-as-state-regularizer** losses that constrain the SSM's internal hidden state to obey the water-balance equation and accumulated growing degree days.

**Pilot scope:** US Corn Belt (Iowa/Illinois), Maize, 2017â€“2023 (Sentinel-2 era only), 1,000â€“3,000 representative 64Ã—64 patches.

**Compute budget:** ~30â€“60 A100-hours total (Google Colab Pro).

---

## 2. Invariants: Rules That Must Never Be Violated

These are hard constraints. If your code violates any of these, stop and fix it before proceeding.

### 2.1 Data Invariants
- **INV-D1:** Coarse-resolution data (ERA5 weather at 31km, SoilGrids at 250m) must NEVER be spatially upsampled to 10m. Each resolution tier produces its own tokens. The model handles multi-resolution through the MLA cross-modal attention block.
- **INV-D2:** No future data leakage. At time step `t`, the model must NOT have access to satellite observations, weather data, or labels from time `t+1` or later. This includes gap-filled products that use future observations.
- **INV-D3:** All normalization statistics (mean, std) must be computed ONLY from the training set. Never include validation or test data in normalization.
- **INV-D4:** The 2023 growing season data is a STRICT extreme-event holdout. It must never appear in any training or validation fold under any circumstance.
- **INV-D5:** Yield predictions are validated at the spatial scale of the labels (county-level for USDA NASS). Aggregate patch-level predictions to county before computing yield metrics.
- **INV-D6:** Irrigation data is treated as a "preferred covariate". If irrigation data is absent for a patch, its features are zeroed, its `modality_present` mask is set to 0, and `I_obs` is treated as 0.0 in the water balance residual. The `I_latent` head must capture all unobserved water inputs.

### 2.2 Architecture Invariants
- **INV-A1:** The SSM block MUST use 4 truly separate SSM sub-modules, each with its own independent A, B, C, D, Î” parameters: `ssm_water`, `ssm_stress`, `ssm_agdd`, `ssm_free`. The input `x_t` is projected into 4 separate streams, each processed by its own SSM recurrence. This ensures each sub-state has genuinely independent recurrent dynamics â€” do NOT use a single monolithic SSM and slice its hidden state. Each sub-state must be independently addressable.
- **INV-A2:** Physics losses are applied to the hidden sub-states, NOT only to output predictions. The water-balance residual is computed from `decode_SM(h_water_t) - decode_SM(h_water_{t-1})`. The AGDD constraint is computed from `decode_AGDD(h_agdd_t)`. The stress asymmetry prior is computed from `decode_stress(h_stress_t) - decode_stress(h_stress_{t-1})`. The `decode_SM` and `decode_AGDD` functions MUST use `Softplus` activation to enforce non-negative outputs (soil moisture and thermal time are non-negative physical quantities). The `decode_stress` function uses `Linear(state_dim, 1)` with NO activation (stress is a latent quantity that can be negative during recovery). All three decode functions are SHARED across all SSM blocks (not per-block). `decode_SM` and `decode_AGDD` are initialized with positive weights (Xavier uniform, then abs()); `decode_stress` uses standard Xavier uniform initialization.
- **INV-A3:** The MoE uses top-1 routing. Only 1 out of 8 experts is active per token per layer.
- **INV-A4:** The backbone interleaves blocks in the pattern: `SSM â†’ SWA â†’ SSM â†’ MLA â†’ SSM â†’ SWA â†’ ...`. SSM blocks always dominate.
- **INV-A5:** The tokenizer's hypernetwork must accept arbitrary channel metadata (wavelength, resolution, variable_id) and produce per-channel projection weights dynamically. It must NOT have hard-coded encoder paths for different modalities. The channel registry MUST include irrigation channels (available via modality-present mask).
- **INV-A6:** The `h_stress` sub-state uses an asymmetry prior (stress accumulates faster than it recovers), NOT a strict monotonicity prior. Plants DO recover from stress. The asymmetry is implemented as: `penalty = mean(relu(-Î”h_stress) * asymmetry_factor)` where `asymmetry_factor > 1` makes decreases costlier than increases but does not forbid them.

### 2.3 Training Invariants
- **INV-T1:** Physics loss lambdas use dynamic warm-up: `lambda(epoch) = lambda_max * min(1, epoch / warmup_epochs)`. Default warmup = 10 epochs.
- **INV-T2:** Modality dropout randomly zeros out entire modalities with p=0.1â€“0.2 during training. This is separate from standard dropout.
- **INV-T3:** Monitor embedding collapse via eigenvalue analysis of the embedding covariance matrix. If top-1 eigenvalue explains > 90% of variance, trigger alert.
- **INV-T4:** Checkpoint the full model state (model, optimizer, scheduler, epoch, best_metric) every epoch. Colab sessions can disconnect without warning.

### 2.4 Terminology Invariants
- **INV-TERM1:** The latent water input variable is called `I_latent`, NEVER "irrigation". It captures ALL unexplained water: irrigation, groundwater, capillary rise, sensor error.
- **INV-TERM2:** The temporal DAG is a "structural prior" or "inductive bias", NEVER "causal discovery".
- **INV-TERM3:** The model is a "research prototype demonstrated at pilot scale", NEVER a "foundation model" or "globally generalizable".

---

## 3. Phase 0: Project Scaffolding

### 3.1 Create the Directory Structure

Create these directories and `__init__.py` files. Execute this exactly:

```
AgroFM/
â”œâ”€â”€ config/
â”‚   â””â”€â”€ AGENTS.md
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ datacube/
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ model/
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ physics/
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ ssl/
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ diagnostics/
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â””â”€â”€ baselines/
â”‚       â””â”€â”€ __init__.py
â”œâ”€â”€ scripts/
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ __init__.py
â””â”€â”€ requirements.txt
```

### 3.2 Create `requirements.txt`

```
torch>=2.1.0
pytorch-lightning>=2.1.0
xarray>=2023.1.0
rasterio>=1.3.0
geopandas>=0.14.0
shapely>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
wandb>=0.16.0
matplotlib>=3.8.0
seaborn>=0.13.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
einops>=0.7.0
```

**Note on mamba-ssm:** The `mamba-ssm` package requires CUDA compilation. If it cannot be installed (e.g., on Colab with incompatible CUDA), implement a pure-PyTorch SSM fallback using the selective scan formulation from the Mamba paper. See Phase 2B for details.

### 3.3 Verify Scaffolding

```bash
python -c "import src; print('src package OK')"
python -c "import src.datacube; import src.model; import src.physics; import src.ssl; import src.diagnostics; import src.baselines; print('All subpackages OK')"
```

Both commands must print "OK". If not, check that all `__init__.py` files exist.

---

## 4. Phase 1A: Configuration Files

### 4.1 File: `config/data_config.yaml`

This file defines ALL data paths, region definitions, and split parameters. The training pipeline reads this file and nothing else for data configuration.

```yaml
# config/data_config.yaml

project:
  name: "AgroSSM-Pilot"
  version: "3.0"

pilot_region:
  name: "US Corn Belt"
  states: ["Iowa", "Illinois"]
  crop: "maize"
  bbox:  # Bounding box [west, south, east, north] in EPSG:4326
    west: -96.7
    south: 37.0
    east: -87.5
    north: 43.5

temporal:
  start_year: 2017
  end_year: 2023
  growing_season_start_doy: 91   # April 1
  growing_season_end_doy: 304    # October 31
  composite_days: 5               # 5-day composites
  sequence_length: 30             # ~30 composites per growing season

patches:
  size: 64                        # 64x64 pixels
  resolution_m: 10                # 10 meters per pixel
  min_crop_fraction: 0.7          # At least 70% maize pixels
  num_patches: 2000               # Target number of patches
  min_counties: 50                # Cover at least 50 counties

splits:
  train_years: [2017, 2018, 2019, 2020, 2021]
  val_strategy: "spatial_temporal_blocked_kfold"
  num_folds: 5
  test_years: [2022]
  extreme_holdout_years: [2023]
  spatial_block_size_km: 50       # For bootstrap evaluation

data_sources:
  optical:
    product: "Sentinel-2 L2A"
    bands: ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
    band_wavelengths_nm: [490, 560, 665, 842, 1610, 2190]
    cloud_threshold: 0.2          # Max 20% cloud cover per composite
  weather:
    product: "ERA5"
    variables: ["t2m_min", "t2m_max", "t2m_mean", "total_precipitation",
                "u10", "v10", "d2m", "ssrd", "pet"]
    spatial_resolution_km: 31
  soil:
    product: "SoilGrids"
    variables: ["clay", "sand", "soc", "phh2o", "cec", "bdod", "ocd"]
    spatial_resolution_m: 250
  crop_mask:
    product: "USDA CDL"
    maize_class_id: 1
  yield_labels:
    product: "USDA NASS"
    level: "county"
    variable: "YIELD"
    commodity: "CORN"
  irrigation:
    product: "WorldCereal"       # WorldCereal for 2021+; GMIA as static fallback
    fallback: "GMIA"             # Global Map of Irrigation Areas (static)
    preferred_covariate: true    # Use when available; modality mask handles absence
    spatial_resolution_m: 10     # WorldCereal at 10m; GMIA at ~10km (patch-level)

normalization:
  strategy: "per_channel_standardization"  # z-score: (x - mean) / std
  compute_from: "train_only"               # INV-D3: never include val/test
```

### 4.2 File: `config/model_config.yaml`

```yaml
# config/model_config.yaml

model:
  name: "AgroSSM"
  embed_dim: 256                  # Token embedding dimension
  
  tokenizer:
    hypernetwork_hidden: 128      # Hidden dim of metadata-to-weights MLP
    hypernetwork_layers: 2        # Number of MLP layers in hypernetwork
    num_channels_max: 26          # Max distinct input channels (25 base + 1 irrigation)
    
  backbone:
    num_blocks: 12                # Total blocks (SSM + SWA + MLA)
    # Interleave pattern: SSM, SWA, SSM, MLA, SSM, SWA, SSM, MLA, SSM, SWA, SSM, MLA
    # That's 6 SSM, 3 SWA, 3 MLA for 12 blocks
    ssm:
      state_dim: 64               # SSM recurrent state dimension per sub-state
      # Total state = 4 * state_dim = 256 (h_water, h_stress, h_agdd, h_free)
      expand_factor: 2            # Mamba expansion factor
      dt_rank: "auto"             # Discretization rank
    swa:
      window_size: 5              # Attend to 5 nearest time steps
      num_heads: 8
      global_every_n: 4           # Every 4th SWA block uses full attention
    mla:
      latent_dim: 32              # Compressed KV dimension
      num_heads: 8

  moe:
    num_experts: 8
    expert_hidden_dim: 512        # Each expert FFN hidden dim
    top_k: 1                      # Top-1 routing
    load_balance_weight: 0.01     # Auxiliary load-balance loss weight

  heads:
    primary:
      stress:
        type: "binary_classification"
        hidden_dim: 128
        output_dim: 1             # Stress probability
      phenology:
        type: "regression"
        hidden_dim: 128
        output_dim: 1             # Days until next transition
      yield_risk:
        type: "regression"
        hidden_dim: 128
        output_dim: 1             # Yield anomaly percentage
    auxiliary:
      drought_trajectory:
        type: "multi_step_regression"
        hidden_dim: 128
        output_dim: 3             # t+1, t+2, t+3 stress trajectory
      residual_water:
        type: "regression_with_uncertainty"
        hidden_dim: 128
        output_dim: 2             # I_latent mean + log_variance

training:
  optimizer: "AdamW"
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: "cosine_warmup"
  warmup_epochs: 5
  max_epochs: 100
  batch_size: 16
  gradient_accumulation_steps: 4  # Effective batch = 64
  gradient_clip_norm: 1.0
  mixed_precision: "bf16"         # BF16 mixed precision
  
  modality_dropout: 0.15          # INV-T2
  
  physics:
    water_balance_lambda_max: 0.1
    agdd_lambda_max: 0.1
    stress_asymmetry_lambda_max: 0.05  # INV-A6: asymmetry prior on h_stress
    stress_asymmetry_factor: 3.0       # Decreases penalized 3× more than increases
    dag_lambda_max: 0.05
    warmup_epochs: 10             # INV-T1
    
  ssl:
    ssl_a_weight: 1.0
    ssl_b_weight: 0.5
    ssl_c_weight: 0.5
    mask_ratio_spatial: 0.75      # 75% of patches masked per time step
    mask_ratio_temporal: 0.20     # 20% of time steps fully masked
    
  moe_load_balance_weight: 0.01
  calibration_weight: 0.01

checkpoint:
  save_every_epoch: true          # INV-T4
  save_dir: "checkpoints/"
  monitor_metric: "val/stress_auroc"
  mode: "max"
```

---

## 5. Phase 1B: Data Pipeline â€” Datacube and Patch Extraction

### 5.1 File: `src/datacube/channel_registry.py`

**Purpose:** Central registry of all input channels with their metadata. The tokenizer's hypernetwork reads this metadata to generate per-channel projection weights.

**Why this file exists:** The unified tokenizer needs to know the physical properties of each input channel (wavelength, resolution, variable type). Hard-coding this inside the tokenizer would violate INV-A5. This registry provides a single source of truth.

```python
"""
Channel registry for the unified tokenizer.

Each channel has:
- channel_id (str): Unique identifier
- channel_type (str): "optical" | "weather" | "soil" | "calendar" | "irrigation"
- wavelength_nm (float or None): Central wavelength for optical bands; None for non-optical
- spatial_resolution_m (float): Native spatial resolution in meters
- temporal_resolution (str): "5day" | "daily" | "static"
- variable_id (int): Integer encoding for the hypernetwork input
"""
from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class ChannelMeta:
    channel_id: str
    channel_type: str
    wavelength_nm: Optional[float]
    spatial_resolution_m: float
    temporal_resolution: str
    variable_id: int

# The complete channel registry for the pilot.
# variable_id values are arbitrary but must be unique and consistent.
CHANNEL_REGISTRY: List[ChannelMeta] = [
    # --- Sentinel-2 optical bands (fine resolution, 10m, 5-day) ---
    ChannelMeta("S2_Blue",  "optical",  490.0,    10.0, "5day", 0),
    ChannelMeta("S2_Green", "optical",  560.0,    10.0, "5day", 1),
    ChannelMeta("S2_Red",   "optical",  665.0,    10.0, "5day", 2),
    ChannelMeta("S2_NIR",   "optical",  842.0,    10.0, "5day", 3),
    ChannelMeta("S2_SWIR1", "optical",  1610.0,   10.0, "5day", 4),
    ChannelMeta("S2_SWIR2", "optical",  2190.0,   10.0, "5day", 5),

    # --- ERA5 weather (coarse, 31km, 5-day aggregated) ---
    ChannelMeta("ERA5_Tmin",   "weather", None, 31000.0, "5day", 6),
    ChannelMeta("ERA5_Tmax",   "weather", None, 31000.0, "5day", 7),
    ChannelMeta("ERA5_Tmean",  "weather", None, 31000.0, "5day", 8),
    ChannelMeta("ERA5_Precip", "weather", None, 31000.0, "5day", 9),
    ChannelMeta("ERA5_Wind_U", "weather", None, 31000.0, "5day", 10),
    ChannelMeta("ERA5_Wind_V", "weather", None, 31000.0, "5day", 11),
    ChannelMeta("ERA5_Dewpt",  "weather", None, 31000.0, "5day", 12),
    ChannelMeta("ERA5_Solar",  "weather", None, 31000.0, "5day", 13),
    ChannelMeta("ERA5_PET",    "weather", None, 31000.0, "5day", 14),

    # --- SoilGrids (medium, 250m, static) ---
    ChannelMeta("Soil_Clay",    "soil", None, 250.0, "static", 15),
    ChannelMeta("Soil_Sand",    "soil", None, 250.0, "static", 16),
    ChannelMeta("Soil_OrgC",    "soil", None, 250.0, "static", 17),
    ChannelMeta("Soil_pH",      "soil", None, 250.0, "static", 18),
    ChannelMeta("Soil_CEC",     "soil", None, 250.0, "static", 19),
    ChannelMeta("Soil_BulkDen", "soil", None, 250.0, "static", 20),
    ChannelMeta("Soil_Depth",   "soil", None, 250.0, "static", 21),

    # --- Crop calendar (patch-level, seasonal) ---
    ChannelMeta("CropType",     "calendar", None, 10.0, "static", 22),
    ChannelMeta("PlantingDOY",  "calendar", None, 10.0, "static", 23),
    ChannelMeta("HarvestDOY",   "calendar", None, 10.0, "static", 24),

    # --- Irrigation (fine resolution when available, modality-present mask handles absence) ---
    ChannelMeta("Irrigation",   "irrigation", None, 10.0, "static", 25),
]

def get_metadata_tensor(channel_ids: List[str]) -> "torch.Tensor":
    """
    Given a list of channel_id strings, return a float tensor of shape
    [num_channels, 4] where columns are:
      [wavelength_nm (0 if None), spatial_resolution_m, temporal_code, variable_id]
    
    temporal_code: 0 = "5day", 1 = "daily", 2 = "static"
    """
    import torch
    temporal_map = {"5day": 0.0, "daily": 1.0, "static": 2.0}
    registry_map = {c.channel_id: c for c in CHANNEL_REGISTRY}
    rows = []
    for cid in channel_ids:
        c = registry_map[cid]
        rows.append([
            c.wavelength_nm if c.wavelength_nm is not None else 0.0,
            c.spatial_resolution_m,
            temporal_map[c.temporal_resolution],
            float(c.variable_id),
        ])
    return torch.tensor(rows, dtype=torch.float32)
```

**Test:** `pytest tests/test_datacube.py -k test_channel_registry`
- Assert `len(CHANNEL_REGISTRY) == 26` (25 base + 1 irrigation)
- Assert all `variable_id` values are unique
- Assert `get_metadata_tensor(["S2_Blue", "ERA5_Precip", "Irrigation"])` returns shape `[3, 4]`

### 5.2 File: `src/datacube/patch_dataset.py`

**Purpose:** PyTorch Dataset class that loads one patch's multi-resolution time series and returns properly shaped tensors.

**Critical design points:**
- Returns a **dictionary** of tensors, NOT a single concatenated tensor. Each resolution tier is a separate entry. The tokenizer handles fusion.
- Includes `modality_present_mask` for every channel at every time step.
- Applies modality dropout during training (INV-T2).

**Return dictionary schema (exact keys and shapes):**

```python
{
    # Fine-resolution optical (satellite)
    "optical": torch.Tensor,           # shape: [T, C_opt, H, W] = [30, 6, 64, 64]
    "optical_mask": torch.Tensor,      # shape: [T, C_opt] = [30, 6] â€” 1.0 if present, 0.0 if cloudy/missing
    
    # Coarse-resolution weather
    "weather": torch.Tensor,           # shape: [T, C_weather] = [30, 9]
    "weather_mask": torch.Tensor,      # shape: [T, C_weather] = [30, 9]
    
    # Static soil properties
    "soil": torch.Tensor,              # shape: [C_soil] = [7]
    "soil_mask": torch.Tensor,         # shape: [C_soil] = [7]
    
    # Crop calendar
    "calendar": torch.Tensor,          # shape: [C_cal] = [3] â€” crop_type, planting_doy, harvest_doy
    "calendar_mask": torch.Tensor,     # shape: [C_cal] = [3]
    
    # Irrigation (preferred covariate; may be absent â€” modality_present handles this)
    "irrigation": torch.Tensor,        # shape: [H, W] = [64, 64] â€” binary classification (0/1) or continuous
    "irrigation_mask": torch.Tensor,   # shape: [1] â€” 1.0 if irrigation map available, 0.0 if absent
    
    # Metadata
    "patch_id": str,
    "county_fips": str,
    "year": int,
    "doy_sequence": torch.Tensor,      # shape: [T] â€” day-of-year for each composite
    "lat": float,
    "lon": float,
    
    # Labels (None during SSL pretraining)
    "stress_label": Optional[torch.Tensor],      # shape: [T, H', W'] = [T, 4, 4] â€” binary stress per spatial token per time step
    "phenology_label": Optional[torch.Tensor],   # shape: [T] — days until next transition
    "yield_anomaly": Optional[float],            # county-level yield anomaly (%)
}
```

**Implementation guidance:**
1. Data files should be stored as `.npz` or `.zarr` per patch, pre-extracted during Sprint 1.3.
2. The `__getitem__` method loads one patch, applies normalization using training-set statistics (INV-D3), applies modality dropout if `self.training` (INV-T2), and returns the dictionary.
3. The `collate_fn` must handle the dictionary structure and pad any variable-length sequences.

**Test:** `pytest tests/test_datacube.py -k test_patch_dataset`
- Load one patch. Assert all tensor shapes match the schema above.
- Assert `optical_mask` has some zeros (cloud-masked composites exist).
- Assert no NaN or Inf in any returned tensor.
- Enable modality dropout and assert at least one modality is fully zeroed in >5% of samples over 100 draws.

### 5.3 File: `src/datacube/split_generator.py`

**Purpose:** Generate the 5-fold Spatial-Temporal Blocked Cross-Validation indices. Ensures zero leakage (INV-D2, INV-D4).

**Algorithm (step by step):**

1. Load the patch metadata index (patch_id, county_fips, year, lat, lon).
2. Filter to training years only: [2017, 2018, 2019, 2020, 2021].
3. Assign each patch to a spatial block based on its (lat, lon) using a 50km grid.
4. For each fold k âˆˆ {0, 1, 2, 3, 4}:
   a. Hold out ~20% of spatial blocks as the validation set.
   b. Ensure no spatial block appears in both train and validation for the same fold.
   c. Ensure temporal diversity: each fold's validation set should span multiple years.
5. Output: a dictionary `{fold_id: {"train_indices": [...], "val_indices": [...]}}`
6. Separately, create `test_indices` (all patches from 2022) and `extreme_indices` (all patches from 2023).

**Test:** `pytest tests/test_datacube.py -k test_split_generator`
- Assert no patch_id appears in both train and val within any fold.
- Assert no patch from 2023 appears in train or val.
- Assert each fold has approximately equal validation set size (Â±10%).

---

## 6. Phase 2A: Unified Wavelength-Conditioned Tokenizer

### 6.1 File: `src/model/tokenizer.py`

**Purpose:** The single encoder for all modalities. This is the replacement for v2.3's four separate encoders + fusion block.

**Architecture:**

```
Input channel value(s) + ChannelMeta
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Metadata Hypernetwork      â”‚
â”‚ MLP: [4] â†’ [128] â†’ [128]  â”‚
â”‚ â†’ [embed_dim Ã— input_dim]  â”‚
â”‚ (generates projection W,b) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼ (W, b = generated weights and bias)
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Dynamic Linear Projection  â”‚
â”‚ token = W @ channel_input  â”‚
â”‚         + b                â”‚
â”‚ + LayerNorm + GELU         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Positional Encoding        â”‚
â”‚ Add: spatial_pos + temp_posâ”‚
â”‚ + resolution_encoding      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
  token (shape: [embed_dim])
```

**Key class: `UnifiedTokenizer`**

```python
class UnifiedTokenizer(nn.Module):
    """
    Unified Wavelength-Conditioned Tokenizer.
    
    Tokenizes ALL input modalities (optical, weather, soil, calendar) through
    a single shared mechanism. Each channel's projection weights are dynamically
    generated by a hypernetwork conditioned on the channel's physical metadata
    (wavelength, spatial resolution, temporal resolution, variable ID).
    
    Args:
        embed_dim (int): Output token embedding dimension. Default: 256.
        hypernetwork_hidden (int): Hidden dimension of the metadata MLP. Default: 128.
        hypernetwork_layers (int): Number of MLP layers. Default: 2.
        max_channels (int): Maximum number of distinct channels. Default: 26.
    
    Forward signature:
        Input: Dict[str, torch.Tensor] â€” the patch dictionary from PatchDataset
        Output: Tuple[torch.Tensor, torch.Tensor]
            - tokens: [B, N_total, embed_dim] â€” all modality tokens concatenated
            - token_masks: [B, N_total] â€” 1.0 if token is real, 0.0 if missing-modality placeholder
    
    Where N_total = N_optical + N_weather + N_soil + N_calendar:
        - N_optical = T * num_spatial_tokens_per_timestep (e.g., T * 16 for 4x4 patches from 64x64)
        - N_weather = T * 1 (one weather token per time step, since weather covers entire patch)
        - N_soil = 1 (static, one token for entire sequence)
        - N_calendar = 1 (static, one token for entire sequence)
    """
```

**Implementation notes:**
- For optical bands: patchify the 64Ã—64 image into 16Ã—16 patches â†’ 4Ã—4 = 16 spatial tokens per time step. Each token is formed by projecting the 6-band values of one 16Ã—16 patch using the hypernetwork-generated weights for each band, then summing. Result: `T Ã— 16` optical tokens.
- For weather: one token per time step. Project the 9 weather variables using their hypernetwork-generated weights and sum. Result: `T Ã— 1` weather tokens.
- For soil: one token (static). Project the 7 soil variables and sum. Result: `1` soil token, broadcast across all time steps.
- For calendar: one token (static). Result: `1` calendar token.
- Missing modalities: replace with a learned `nn.Parameter(torch.randn(embed_dim))` placeholder token. Set the corresponding mask position to 0.0.

**Test:** `pytest tests/test_model.py -k test_tokenizer`
- Create a dummy batch of 2 patches.
- Assert output `tokens` shape is `[2, N_total, 256]`.
- Zero out the weather modality. Assert the weather token positions in `token_masks` are 0.0.
- Assert no NaN in output.

---

## 7. Phase 2B: Hybrid Temporal Backbone (SSM + SWA + MLA)

### 7.1 File: `src/model/ssm_block.py`

**Purpose:** The State-Space Model block with truly partitioned physical sub-modules. This is the CORE NOVELTY of AgroSSM.

**CRITICAL DESIGN DECISION (C2 Resolution): Truly Partitioned SSM Sub-Modules**

A monolithic Mamba block treats its hidden state as a single vector updated by shared A, B, C, D matrices. Slicing the output and calling the slices `h_water`, `h_agdd`, etc. is architecturally meaningless â€” the sub-slices are entangled through shared matrix multiplications and do not have independent recurrent dynamics.

**AgroSSM uses 4 genuinely independent SSM sub-modules**, each with its own recurrence parameters:

```python
class PhysicalSSMBlock(nn.Module):
    """
    SSM block with 4 truly independent sub-modules, each with its own
    A, B, C, D, Î” parameters and its own hidden state.
    
    Architecture:
        x_t [embed_dim] â†’ project into 4 streams of dim [state_dim] each
                         â†’ ssm_water(stream_0) â†’ h_water_t
                         â†’ ssm_stress(stream_1) â†’ h_stress_t
                         â†’ ssm_agdd(stream_2)  â†’ h_agdd_t
                         â†’ ssm_free(stream_3)  â†’ h_free_t
                         â†’ concatenate + project back to [embed_dim]
    
    Each sub-SSM has:
        - Its own A matrix (state_dim Ã— state_dim, diagonal or full)
        - Its own B, C projection functions (input-dependent, Mamba-style)
        - Its own Î” discretization (input-dependent, Mamba-style)
        - Its own D skip connection
    
    This ensures h_water evolves according to its own learned dynamics
    (regularized by the water-balance equation) independently from h_agdd
    (regularized by the AGDD constraint).
    
    Args:
        embed_dim (int): Input/output token dimension. Default: 256.
        state_dim (int): Dimension of EACH sub-state. Default: 64.
        expand_factor (int): Internal expansion for each sub-SSM. Default: 2.
    
    Forward:
        Input: x of shape [B, T, embed_dim]
        Output: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - output: [B, T, embed_dim]
            - substates: {"h_water": [B, T, state_dim], "h_stress": [B, T, state_dim],
                          "h_agdd": [B, T, state_dim], "h_free": [B, T, state_dim]}
    """
    def __init__(self, embed_dim=256, state_dim=64, expand_factor=2):
        super().__init__()
        # Project input into 4 streams
        self.input_proj = nn.Linear(embed_dim, 4 * state_dim)
        
        # 4 independent SSM sub-modules
        self.ssm_water  = SelectiveScanSSM(state_dim, expand_factor)  # water-balance dynamics
        self.ssm_stress = SelectiveScanSSM(state_dim, expand_factor)  # stress accumulation dynamics
        self.ssm_agdd   = SelectiveScanSSM(state_dim, expand_factor)  # thermal accumulation dynamics
        self.ssm_free   = SelectiveScanSSM(state_dim, expand_factor)  # unconstrained dynamics
        
        # Project back to embed_dim
        self.output_proj = nn.Linear(4 * state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
```

**The `SelectiveScanSSM` sub-module (pure PyTorch, no CUDA dependency):**

```python
class SelectiveScanSSM(nn.Module):
    """
    A single selective-scan SSM sub-module implementing:
        delta_t = softplus(linear_dt(x_t))
        A_bar = exp(A * delta_t)           # A is diagonal, learnable, initialized negative
        B_bar = delta_t * B(x_t)           # B is input-dependent
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C(x_t) * h_t + D * x_t      # C is input-dependent
    
    This runs sequentially (O(T * state_dim)) and requires no custom CUDA kernels.
    
    Args:
        dim (int): Input/output/state dimension.
        expand_factor (int): Internal expansion. Default: 2.
    """
```

**Why this design is defensible:** When a reviewer asks "how do you know `h_water` actually tracks water dynamics?", the answer is: "It has its own independent recurrence with its own learned A, B, C, D parameters, and the only external signal shaping its dynamics is the water-balance residual loss. There is no parameter sharing with `h_agdd` that could contaminate the water signal."

**The sub-state must be returned at every time step** so the physics regularizer can access it. Do NOT discard intermediate states.

**Test:** `pytest tests/test_model.py -k test_ssm_block`
- Forward pass with shape `[2, 30, 256]` â†’ output shape `[2, 30, 256]`.
- Assert `substates["h_water"]` shape is `[2, 30, 64]`.
- Assert all 4 sub-states are truly independent: freezing `ssm_water` parameters and backpropagating through `h_agdd` must NOT produce gradients in `ssm_water`.
- Assert no NaN.

### 7.2 File: `src/model/swa_block.py`

**Purpose:** Sliding-Window Attention for local spatio-temporal patterns.

```python
class SlidingWindowAttentionBlock(nn.Module):
    """
    Local sliding-window multi-head attention.
    
    Each token attends only to tokens within a window of `window_size` positions.
    Every `global_every_n`-th instance uses full (global) attention instead.
    
    Args:
        embed_dim (int): Token dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        window_size (int): Number of neighboring positions to attend to. Default: 5.
        is_global (bool): If True, use full attention (no windowing). Default: False.
    
    Forward:
        Input: x [B, N, embed_dim], mask [B, N] (1=real, 0=pad)
        Output: x [B, N, embed_dim]
    """
```

**Implementation:** Use `torch.nn.MultiheadAttention` with a custom `attn_mask` that restricts attention to a diagonal band of width `window_size`. For global layers, pass no mask.

**Test:** `pytest tests/test_model.py -k test_swa_block`
- Forward pass `[2, 30, 256]` â†’ `[2, 30, 256]`.
- Assert that with `window_size=5`, attention weights are zero outside a 5-wide band (inspect `attn_weights`).

### 7.3 File: `src/model/mla_block.py`

**Purpose:** Multi-Head Latent Attention for cross-resolution context compression.

```python
class MultiHeadLatentAttention(nn.Module):
    """
    Compresses coarse Key-Value pairs into a low-dimensional latent space,
    then fine-resolution queries attend to this compressed context.
    
    Used for: letting 10m satellite tokens query 31km weather context
    without spatially upsampling the weather data.
    
    Args:
        embed_dim (int): Query/output dimension. Default: 256.
        latent_dim (int): Compressed KV dimension. Default: 32.
        num_heads (int): Number of attention heads. Default: 8.
    
    Forward:
        Input:
            fine_tokens: [B, N_fine, embed_dim]   â€” satellite tokens (queries)
            coarse_tokens: [B, N_coarse, embed_dim] â€” weather/soil tokens (keys/values)
            coarse_mask: [B, N_coarse] â€” 1=real, 0=missing
        Output: [B, N_fine, embed_dim]
    """
```

**Implementation:**
1. Project coarse tokens to latent KV: `latent_kv = linear_compress(coarse_tokens)` â†’ `[B, N_coarse, latent_dim]`
2. Compute attention: `out = multi_head_attention(query=fine_tokens, key=latent_kv, value=latent_kv)`
3. Project back: `out = linear_expand(out)` â†’ `[B, N_fine, embed_dim]`
4. Residual connection + LayerNorm.

**Test:** `pytest tests/test_model.py -k test_mla_block`
- Fine tokens `[2, 480, 256]`, coarse tokens `[2, 30, 256]` â†’ output `[2, 480, 256]`.

### 7.4 File: `src/model/backbone.py`

**Purpose:** Assembles the hybrid backbone by interleaving SSM, SWA, and MLA blocks.

```python
class HybridTemporalBackbone(nn.Module):
    """
    Interleaves SSM, SWA, and MLA blocks in the pattern:
        SSM â†’ SWA â†’ SSM â†’ MLA â†’ SSM â†’ SWA â†’ SSM â†’ MLA â†’ ...
    
    INV-A4: SSM blocks dominate. The pattern repeats to fill num_blocks.
    
    Forward:
        Input:
            fine_tokens: [B, N_fine, embed_dim] â€” optical tokens
            coarse_tokens: [B, N_coarse, embed_dim] â€” weather+soil+calendar tokens
            fine_mask: [B, N_fine]
            coarse_mask: [B, N_coarse]
        Output: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - output: [B, N_fine, embed_dim] â€” processed fine tokens
            - all_substates: {"h_water": [B, T, state_dim], ...} â€” from last SSM block
    """
```

**Implementation:** Maintain two token streams: `fine` and `coarse`. SSM processes `fine`. SWA processes `fine`. MLA takes `fine` as queries and `coarse` as keys/values. The SSM sub-states are accumulated and returned.

**IMPORTANT:** The sub-states from the LAST SSM block are the ones passed to the physics regularizer and output heads. Intermediate SSM sub-states are discarded to save memory (but this is an implementation choice â€” if memory allows, keeping all gives richer physics signal).

**Test:** `pytest tests/test_model.py -k test_backbone`
- Full forward pass. Assert output shape and substate shapes.
- Assert the block count matches config.

---

## 8. Phase 2C: Regime-Routed Mixture-of-Experts

### 8.1 File: `src/model/moe.py`

```python
class RegimeRoutedMoE(nn.Module):
    """
    Top-1 routed Mixture-of-Experts with load-balance loss.
    
    Each expert is a small FFN: Linear(embed_dim, expert_hidden) â†’ GELU â†’ Linear(expert_hidden, embed_dim).
    The router is a Linear(embed_dim, num_experts) that scores each token against each expert.
    Top-1 selection: each token is processed by exactly one expert.
    
    INV-A3: top_k=1.
    
    Returns:
        output: [B, N, embed_dim]
        load_balance_loss: scalar â€” auxiliary loss to prevent expert collapse
        expert_assignments: [B, N] â€” which expert was selected (for logging)
    """
```

**Load-balance loss formula:**
```python
# f_i = fraction of tokens routed to expert i
# P_i = mean router probability for expert i across all tokens
# load_balance_loss = num_experts * sum(f_i * P_i)
# This penalizes uneven routing.
```

**Test:** `pytest tests/test_model.py -k test_moe`
- Forward pass `[2, 480, 256]` â†’ `[2, 480, 256]`.
- Assert all 8 experts receive at least 1 token (no completely dead experts on a batch of 960 tokens).
- Assert `load_balance_loss` is a positive scalar.

---

## 9. Phase 2D: Output Heads and Physics-as-State-Regularizer

### 9.1 File: `src/model/heads.py`

```python
class CropStressHead(nn.Module):
    """
    Spatial binary classification: P(stress) per spatial token per time step.
    
    The tokenizer creates a 4Ã—4 grid of spatial tokens from each 64Ã—64 patch.
    The stress head outputs one stress probability per spatial token, enabling
    spatial stress maps and proper AUROC computation (H4 resolution).
    
    Input: z_t [B, T, N_spatial, embed_dim] where N_spatial = 16 (4Ã—4 grid)
    Output: [B, T, 4, 4] â€” stress probability map at token resolution
    """

class PhenologyHead(nn.Module):
    """Regression: days until next phenological transition. Input: z_t. Output: [B, T, 1]."""

class YieldRiskHead(nn.Module):
    """Regression: yield anomaly %. Input: z_T (last time step). Output: [B, 1]."""

class DroughtTrajectoryHead(nn.Module):
    """Multi-step regression: stress at t+1, t+2, t+3. Input: z_t. Output: [B, T, 3]."""

class ResidualWaterHead(nn.Module):
    """Regression with uncertainty: I_latent mean + log_variance. Input: z_t. Output: [B, T, 2]."""
```

Each head is a 2-layer MLP: `Linear(embed_dim, hidden) â†’ GELU â†’ Linear(hidden, output_dim)`.

**Note on CropStressHead (H4 Resolution):** The spatial output `[B, T, 4, 4]` aligns with the tokenizer's 4Ã—4 grid of spatial tokens. AUROC is computed over this spatial grid across all patches and time steps. The v2.3 "spatial IoU" metric is computed at this token resolution (each token covers 16Ã—16 = ~160m Ã— 160m).

### 9.2 File: `src/physics/state_regularizer.py`

**Purpose:** Implements the physics-as-state-regularizer losses (INV-A2).

```python
class WaterBalanceStateRegularizer(nn.Module):
    """
    Computes water-balance residual on h_water sub-states (H1/H2 unified formula).
    
    UNIFIED FORMULA (consistent across all documents):
    residual_t = decode_SM(h_water_t) - decode_SM(h_water_{t-1})
                 - (P_t + I_obs_t + I_latent_t - ET_t - R_t - D_t)
    L_water = mean(residual_t ** 2)
    
    Where:
      - I_obs_t = observed irrigation (from irrigation map), 0.0 when absent
      - I_latent_t = model-inferred latent residual water input (from ResidualWaterHead)
      - R_t = runoff, estimated via simplified SCS-CN method:
              R = (P - 0.2*S)^2 / (P + 0.8*S)  where S = (25400/CN) - 254
              CN depends on soil type (from SoilGrids) and antecedent moisture
      - D_t = deep drainage, estimated via exponential percolation:
              D = K_sat * exp(-alpha * SM_deficit)
              K_sat from SoilGrids bulk density + sand/clay fractions
    
    The decode_SM function:
      - Architecture: Linear(state_dim, 1) + Softplus activation
      - Enforces non-negative output (soil moisture cannot be negative)
      - Initialized with positive weights (Xavier uniform, then abs())
      - SHARED across all SSM blocks (not per-block)
    
    Inputs:
        h_water: [B, T, state_dim] â€” water sub-state sequence
        precipitation: [B, T] â€” observed precipitation (mm/5-day)
        et: [B, T] â€” evapotranspiration estimate (mm/5-day)
        i_obs: [B, T] â€” observed irrigation (mm/5-day), 0.0 when irrigation data absent
        i_latent: [B, T] â€” model-inferred latent water input (from ResidualWaterHead)
        soil_properties: [B, 7] â€” SoilGrids variables (for R and D estimation)
    
    Output: scalar loss
    """

class AGDDStateRegularizer(nn.Module):
    """
    Computes AGDD constraint on h_agdd sub-states.
    
    AGDD_t = AGDD_{t-1} + max(0, T_mean_t - T_base)     where T_base = 10Â°C for maize
    L_agdd = mean((decode_AGDD(h_agdd_t) - AGDD_t) ** 2) + monotonicity_penalty
    
    monotonicity_penalty = mean(relu(decode_AGDD(h_agdd_{t-1}) - decode_AGDD(h_agdd_t)))
    (penalizes any decrease in thermal accumulation â€” AGDD is strictly non-decreasing)
    
    The decode_AGDD function:
      - Architecture: Linear(state_dim, 1) + Softplus activation
      - Enforces non-negative output (thermal time is non-negative)
      - Initialized with positive weights (Xavier uniform, then abs())
      - SHARED across all SSM blocks (not per-block)
    
    Inputs:
        h_agdd: [B, T, state_dim]
        temperature_mean: [B, T] â€” mean temperature (Â°C)
        planting_doy: [B] â€” day of year when crop was planted
        doy_sequence: [B, T] â€” day-of-year for each time step
    
    Output: scalar loss
    """

class StressAsymmetryRegularizer(nn.Module):
    """
    Computes asymmetry prior on h_stress sub-states (M1 resolution).
    
    Unlike AGDD (which is strictly monotonic), crop stress CAN recover â€”
    plants recover when rain arrives after drought. But recovery is SLOWER
    than stress onset (biological damage takes time to heal).
    
    L_stress_asym = mean(relu(-delta_h_stress) * asymmetry_factor)
    
    Where:
      - delta_h_stress = decode_stress(h_stress_t) - decode_stress(h_stress_{t-1})
      - asymmetry_factor = 3.0 (default; decreases in stress are penalized 3x more than increases)
      - relu(-delta) selects only decreasing steps
    
    The decode_stress function:
      - Architecture: Linear(state_dim, 1) — NO activation (stress is a latent quantity;
        unlike soil moisture/AGDD, stress can be negative during recovery phases)
      - Initialization: Xavier uniform (standard — no abs() needed)
      - SHARED across all SSM blocks (consistent with decode_SM/decode_AGDD pattern)
    
    This allows recovery but makes it costlier than accumulation, reflecting biology.
    
    Inputs:
        h_stress: [B, T, state_dim]
    
    Output: scalar loss
    """
```

**Test:** `pytest tests/test_physics.py`
- For WaterBalance: create synthetic data where P + I_obs + I_latent - ET - R - D = Î”SM exactly. Assert loss â‰ˆ 0.
- For AGDD: create synthetic temperature data and manually compute AGDD. Assert loss â‰ˆ 0 when h_agdd matches.
- Assert monotonicity penalty > 0 when h_agdd decreases between steps.
- Assert decode_SM and decode_AGDD always produce non-negative outputs (test with random negative inputs).
- For StressAsymmetry: assert loss > 0 when stress decreases, loss = 0 when stress increases.

---

## 10. Phase 2E: Full AgroSSM Model Assembly

### 10.1 File: `src/model/agrossm.py`

**Purpose:** Top-level model class that wires everything together.

```python
class AgroSSM(nn.Module):
    """
    AgroSSM: Compute-Efficient Process-Structured Agro-Ecosystem Model.
    
    Data flow:
        1. Tokenizer converts multi-resolution inputs into tokens
        2. Separate tokens into fine (optical) and coarse (weather+soil+calendar)
        3. Backbone processes fine tokens with coarse context
        4. MoE routes each token through a specialist expert
        5. Pool temporal tokens to get z_t at each time step
        6. Output heads produce predictions from z_t
        7. Physics regularizers constrain SSM sub-states
    
    Forward:
        Input: Dict from PatchDataset (see Section 5.2 schema)
        Output: Dict containing:
            - "stress_logits": [B, T, 4, 4]   â€” spatial stress map (H4: spatial output)
            - "phenology_pred": [B, T, 1]
            - "yield_pred": [B, 1]
            - "drought_traj": [B, T, 3]
            - "residual_water": [B, T, 2]     â€” I_latent mean + log_variance
            - "water_balance_loss": scalar
            - "agdd_loss": scalar
            - "stress_asymmetry_loss": scalar  â€” INV-A6 asymmetry prior
            - "moe_load_balance_loss": scalar
            - "substates": Dict[str, Tensor]
    """
```

**Test:** `pytest tests/test_model.py -k test_agrossm_full`
- Create a dummy batch from `PatchDataset`. Forward pass through `AgroSSM`.
- Assert all output keys exist with correct shapes.
- Assert total parameter count is in range [40M, 80M].
- Assert backward pass completes without error (gradients flow).

---

## 11. Phase 3: SSL Pretext Tasks

### 11.1 File: `src/ssl/ssl_a_reconstruction.py`

Masked spatiotemporal reconstruction. Mask 75% of spatial tokens, 20% of full time steps. L1 loss on masked tokens only.

### 11.2 File: `src/ssl/ssl_b_phenology.py`

Phenology transition prediction. Given current state, predict days until next NDVI inflection point. Huber loss.

### 11.3 File: `src/ssl/ssl_c_trajectory.py`

Multi-horizon drought-stress trajectory. Predict NDVI anomaly at t+1, t+2, t+3. MSE with exponential time discounting.

### 11.4 File: `src/ssl/collapse_monitor.py`

Compute eigenvalues of embedding covariance matrix. Alert if effective rank drops below threshold (INV-T3).

---

## 12. Phase 4: Training Pipeline

### 12.1 File: `scripts/train_model.py`

PyTorch Lightning `LightningModule` wrapping `AgroSSM`. Implements the combined loss function:

```python
loss = (ssl_a_weight * ssl_a_loss
      + ssl_b_weight * ssl_b_loss
      + ssl_c_weight * ssl_c_loss
      + lambda_water(epoch) * water_balance_loss
      + lambda_growth(epoch) * agdd_loss
      + lambda_stress_asym(epoch) * stress_asymmetry_loss  # INV-A6
      + lambda_dag(epoch) * dag_loss
      + moe_load_balance_weight * moe_lb_loss
      + calibration_weight * calibration_loss)
```

**Lambda scheduling (INV-T1):**
```python
def get_physics_lambda(epoch: int, lambda_max: float, warmup: int) -> float:
    return lambda_max * min(1.0, epoch / warmup)
```

**Checkpointing (INV-T4):** Save every epoch. Include model state, optimizer state, scheduler state, epoch number, best metric value, and the RNG state for reproducibility.

---

## 13. Phase 5: Baselines

Implement in `src/baselines/`. Each baseline exposes a `fit(X_train, y_train)` and `predict(X_test)` interface.

| File | Model | Compute |
|------|-------|---------|
| `climatology.py` | Historical mean | ~Free |
| `persistence.py` | Predict last observed value | ~Free |
| `gdd_phenology.py` | GDD thermal-time model | ~Free |
| `drought_indices.py` | SPI/SPEI/VHI | ~Free |
| `tree_models.py` | XGBoost on hand-crafted NDVI+weather features | ~1â€“2 CPU-hrs |
| `lstm_tcn.py` | LSTM and TCN on time-series features | ~3â€“6 GPU-hrs |

---

## 14. Phase 6: Evaluation and Diagnostics

### 14.1 File: `src/diagnostics/bootstrap.py`

Spatial Block Bootstrap (50km blocks, 1000 resamples) and Year-Level LOO Bootstrap. Returns 95% CI for all metrics.

### 14.2 File: `src/diagnostics/causal_dag.py`

Temporal DAG sign consistency checker. For each edge in the DAG (e.g., Temperature â†’ AGDD), compute the partial derivative and check its sign against the agronomic prior. Report % consistency.

### 14.3 File: `src/diagnostics/uncertainty.py`

ECE (Expected Calibration Error), CRPS, and PICP computation.

### 14.4 Pilot Scope Limitation: Sensor-Mix OOD (M2 Resolution)

The v2.3 spec defines "Sensor-Mix OOD" (evaluating with different sensor configurations than training). Since the pilot restricts to Sentinel-2 era only (2017â€“2023) and does NOT implement EO harmonization, this OOD dimension is **explicitly out of scope for the pilot**. Document as: "Sensor-Mix OOD evaluation requires EO harmonization across Landsat and Sentinel-2, which is deferred to the full foundation model. See scaling projection (Section 13) for the harmonization path."

A stub file `src/datacube/eo_harmonization.py` is created to document the scaling path but is a no-op for Sentinel-only data.

---

## 15. Failure Recovery Playbook

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **NaN in gradients** | Check `torch.isnan(loss)` before backward | Reduce learning rate by 10Ã—. If persists, disable physics losses. If still persists, check data for NaN. |
| **SSL collapse** | Effective rank < 0.5 Ã— embed_dim | Increase temperature in SSL-A masking. Add small noise to embeddings. If persists, revert to generic MAE. |
| **MoE expert collapse** | Any expert receives < 2% of tokens | Increase `load_balance_weight` by 2Ã—. If persists, re-initialize dead expert weights from the most-used expert + noise. |
| **OOM on A100-40GB** | CUDA out of memory error | Reduce batch_size. If already at 4, reduce sequence_length to 20. If still OOM, reduce embed_dim to 192. |
| **Colab disconnects mid-training** | Session terminated | Resume from latest checkpoint: `trainer.fit(model, ckpt_path="last")`. All state is restored. |
| **Physics loss destabilizes training** | Training loss oscillates or diverges after physics warmup starts | Double the warmup_epochs (10 â†’ 20). If persists, halve lambda_max. If still unstable, remove physics losses and report as negative result. |
| **Yield RÂ² < 0.3** | Test metric far below MVP target (0.50) | Check that predictions are aggregated to county level (INV-D5). Check yield label quality. If data issue, report as limitation. |

---

## 16. File Index â€” Every File You Will Create

Listed in dependency order. Implement files top-to-bottom; each file depends only on files above it.

| # | File | Phase | Dependencies |
|---|------|-------|--------------|
| 1 | `requirements.txt` | 0 | None |
| 2 | `config/data_config.yaml` | 1A | None |
| 3 | `config/model_config.yaml` | 1A | None |
| 4 | `src/datacube/channel_registry.py` | 1B | None |
| 5 | `src/datacube/patch_dataset.py` | 1B | #4 |
| 6 | `src/datacube/split_generator.py` | 1B | #5 |
| 7 | `src/datacube/eo_harmonization.py` | 1B | None (stub for M2; no-op for Sentinel-only pilot) |
| 8 | `tests/test_datacube.py` | 1B | #4, #5, #6 |
| 9 | `src/model/tokenizer.py` | 2A | #4 |
| 10 | `src/model/ssm_block.py` | 2B | None |
| 11 | `src/model/swa_block.py` | 2B | None |
| 12 | `src/model/mla_block.py` | 2B | None |
| 13 | `src/model/backbone.py` | 2B | #10, #11, #12 |
| 14 | `src/model/moe.py` | 2C | None |
| 15 | `src/model/heads.py` | 2D | None |
| 16 | `src/physics/state_regularizer.py` | 2D | None |
| 17 | `src/model/agrossm.py` | 2E | #9, #13, #14, #15, #16 |
| 18 | `tests/test_model.py` | 2E | #9â€“#17 |
| 19 | `tests/test_physics.py` | 2D | #16 |
| 20 | `src/ssl/ssl_a_reconstruction.py` | 3 | #17 |
| 21 | `src/ssl/ssl_b_phenology.py` | 3 | #17 |
| 22 | `src/ssl/ssl_c_trajectory.py` | 3 | #17 |
| 23 | `src/ssl/collapse_monitor.py` | 3 | None |
| 24 | `tests/test_ssl.py` | 3 | #20â€“#23 |
| 25 | `scripts/train_model.py` | 4 | #5, #6, #17, #20â€“#23 |
| 26 | `src/baselines/climatology.py` | 5 | None |
| 27 | `src/baselines/persistence.py` | 5 | None |
| 28 | `src/baselines/gdd_phenology.py` | 5 | None |
| 29 | `src/baselines/drought_indices.py` | 5 | None |
| 30 | `src/baselines/tree_models.py` | 5 | None |
| 31 | `src/baselines/lstm_tcn.py` | 5 | None |
| 32 | `src/diagnostics/bootstrap.py` | 6 | None |
| 33 | `src/diagnostics/causal_dag.py` | 6 | None |
| 34 | `src/diagnostics/uncertainty.py` | 6 | None |
| 35 | `scripts/evaluate_model.py` | 6 | #17, #32â€“#34 |
| 36 | `tests/test_diagnostics.py` | 6 | #32â€“#34 |

**Total: 36 files.** Every one is specified above with its purpose, interface, and test criteria.

---

## 17. Revision History

| Revision | Date | Issues Resolved | Description |
|----------|------|-----------------|-------------|
| v1.0 | 2026-06-11 | Initial | Original agent handoff document |
| v1.1 | 2026-06-11 | C1, C2, H1, H2, H3, H4, M1, M2 | Applied 8 revisions from rigorous validation review. See below for details. |

### v1.1 Revision Details

| ID | Priority | What Changed |
|----|----------|--------------|
| C1 | CRITICAL | Added `Irrigation` channel (variable_id=25) to CHANNEL_REGISTRY. Added `irrigation`/`irrigation_mask` keys to PatchDataset. Bumped `num_channels_max` to 26. Added irrigation data source to data_config.yaml. |
| C2 | CRITICAL | Rewrote SSM block from monolithic Mamba with sliced hidden state to 4 truly independent `SelectiveScanSSM` sub-modules (`ssm_water`, `ssm_stress`, `ssm_agdd`, `ssm_free`), each with its own A, B, C, D, Î” parameters. This ensures sub-states have genuinely independent recurrent dynamics. |
| H1 | HIGH | Unified water-balance formula across all documents to: `residual = Î”SM - (P + I_obs + I_latent - ET - R - D)`. `I_obs` is now included (0.0 when irrigation data absent). |
| H2 | HIGH | Added explicit R (runoff via simplified SCS-CN) and D (deep drainage via exponential percolation) computation to `WaterBalanceStateRegularizer`, using SoilGrids soil properties as inputs. |
| H3 | HIGH | Specified `decode_SM` and `decode_AGDD`: `Linear(state_dim, 1) + Softplus` activation for non-negativity; initialized with positive weights; SHARED across all SSM blocks. |
| H4 | HIGH | Changed CropStressHead output from `[B, T, 1]` to `[B, T, 4, 4]` (spatial stress map at token resolution). Updated stress_label shape in PatchDataset to match. |
| M1 | MEDIUM | Replaced `h_stress` monotonicity prior with `StressAsymmetryRegularizer` â€” stress can recover but decreases are penalized 3Ã— more than increases, reflecting biological asymmetry. Added INV-A6. |
| M2 | MEDIUM | Added `src/datacube/eo_harmonization.py` stub to file index. Explicitly scoped Sensor-Mix OOD out of pilot claims. Documented as future work requiring EO harmonization. |

---

*End of AgroSSM Agent Handoff Document (v1.1 â€” Post-Validation Review).*
