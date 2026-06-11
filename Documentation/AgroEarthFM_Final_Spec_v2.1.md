# Spec: AgroEarthFM-MVP Research Prototype v2.1

## 1. Objective
Build a multimodal, spatiotemporally sparse, physics-regularized, and causally-informed machine learning research prototype for crop stress detection, crop phenology forecasting, and agricultural yield-risk/anomaly prediction. 

The goal is to deliver a Year-1 Minimum Viable Research Contribution (MVRC) that demonstrates:
1. **C1 (Joint Modeling):** Joint drought + phenology representation improves performance over single-task baselines.
2. **C2 (Multimodal Fusion):** Multi-source input improves out-of-distribution (OOD) robustness.
3. **C3 (Process SSL):** Process-centered Self-Supervised Learning outperforms general masked autoencoders.
4. **C4 (Physics Regularization):** Weak water-balance constraints reduce physical inconsistencies without degrading metric performance.
5. **C5 (Causal Diagnostics):** Structured temporal Directed Acyclic Graph (DAG) constraints provide biologically plausible representations.
6. **C6 (Tail Events):** Robustness under extreme weather events and spatial-temporal block holdouts.

---

## 2. Tech Stack & Dependencies
- **Programming Language:** Python 3.10+
- **Core ML Framework:** PyTorch 2.1+ / PyTorch Lightning 2.1+
- **Data Processing:** Xarray, Rasterio, Pandas, NumPy, Scipy
- **Spatial Analysis:** Geopandas, Shapely
- **Logging & Visualization:** Weights & Biases, Matplotlib, Seaborn
- **Testing & Quality:** PyTest, Flake8, Black, MyPy

---

## 3. Commands
- **Environment Setup:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
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

## 4. Project Structure
```text
agro_fm_mvp/
├── config/
│   ├── data_config.yaml            # Data paths, crop regions, variables
│   └── model_config.yaml            # Hyperparameters, head configs, losses
├── Documentation/
│   └── AgroEarthFM_Final_Spec_v2.1.md # This specification document
├── scripts/
│   ├── preprocess_data.py          # Sparsity extraction and datacube builder
│   ├── train_model.py              # Main training script (PyTorch Lightning)
│   └── evaluate_model.py           # Evaluation, bootstrapping, and causal audits
├── src/
│   ├── datacube/
│   │   ├── __init__.py
│   │   ├── datacube_loader.py      # Spatial-temporal patch dataset loading
│   │   └── patch_extractor.py      # 64x64 representative agricultural patch extractor
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── satellite_encoder.py    # ViT-based Sentinel-2/Landsat encoder
│   │   ├── weather_encoder.py      # Temporal weather transformer encoder
│   │   ├── soil_encoder.py         # Static MLP encoder for soil/topography
│   │   └── calendar_encoder.py     # Crop stage prior embedding
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── cross_attention.py      # Scale-aware cross-attention layer
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── primary_heads.py        # Crop Stress, Phenology, Yield Risk heads
│   │   └── auxiliary_heads.py      # Drought Trajectory, Residual Water/Uncertainty heads
│   ├── physics/
│   │   ├── __init__.py
│   │   └── constraints.py          # Weak water-balance and AGDD growth losses
│   └── diagnostics/
│       ├── __init__.py
│       ├── causal_dag.py           # Temporal DAG constraint and sign auditing
│       └── bootstrap.py            # Spatial block bootstrap & Year LOO bootstrap
└── tests/
    ├── __init__.py
    ├── test_datacube.py            # Unit tests for patch loading and missing modalities
    ├── test_encoders.py            # Dimension and tensor flow tests
    ├── test_physics.py             # Validation of water-balance constraints
    └── test_diagnostics.py         # Autocyclicity and gradient leak tests
```

---

## 5. Architectural Specifications (Resolved from Grill-Me)

### 5.1 Output Heads
To resolve the inconsistencies, the network outputs are divided into:
1. **Primary Supervised Downstream Heads:**
   - **Crop Stress Head:** Output probability of crop stress (water/heat) and spatial IoU.
   - **Phenology Forecast Head:** Output expected timing of next phenological transition (MAE in days).
   - **Yield-Risk Head:** Predict district/regional yield anomaly percentage (RMSE target < 10%).
2. **Auxiliary Diagnostic Heads:**
   - **Drought Trajectory Head:** Regress short-to-medium term drought progression anomalies.
   - **Residual Water/Uncertainty Head:** Estimate predictive uncertainty and infer the latent unobserved water input ($I_{\text{latent}}$).

### 5.2 Direct Irrigation Inputs
- **Policy:** The model **requires** irrigation maps as direct inputs. It is restricted to regions where irrigation maps are available. Modality masks and learned missing-modality placeholder tokens remain active for other inputs (e.g. soil moisture or satellite dropouts), but irrigation data is treated as a baseline required covariate to avoid over-reliance on $I_{\text{latent}}$ inference.

### 5.3 Scope, Spatiotemporal Sparsity, & Regional Matrix
- **Temporal Coverage:** 2000–2026.
- **Crop-Region Matrix (Balanced & De-confounded):**
  - **Wheat:** Indo-Gangetic Plain (India), Iberian Peninsula (Spain)
  - **Maize:** US Corn Belt (Iowa/Illinois), Po Valley (Italy)
- **Sparsity Optimization:** To reduce the 40+ TB storage and 50k+ GPU-hour requirements, training utilizes **Spatiotemporal Sparsity**. The system extracts and trains only on $1,000\text{--}5,000$ representative $64 \times 64$ agricultural patches across each region. The code structure remains flexible (modular interfaces) to swap this with frozen encoders or MODIS coarse-to-fine pre-training.

### 5.4 Data Splits & Temporal Blocked K-Fold
- **Chronological Split:**
  - **Train/Validation Pool:** 2000–2021
  - **In-Region Test set:** 2022–2025
  - **Extreme Drought Years (Held out entirely for extreme-event evaluation):** 2003 (Spain), 2012 (US Corn Belt), 2022 (IGP)
- **Validation Protocol:** Within the 2000–2021 pool, hyperparameters are tuned using **5-Fold Spatial-Temporal Blocked Cross-Validation** to eliminate spatial and temporal leakage.

### 5.5 Autocorrelation & Statistical Protocol
- Naive pixel-level random splits and t-tests are prohibited.
- Performance statistics (confidence intervals, p-values) are calculated using a **Spatial Block Bootstrap ($50\text{--}\text{km} \times 50\text{--}\text{km}$ blocks)** and a **Year-Level Leave-One-Out (LOO) Bootstrap**.

### 5.6 Data Licensing & Access Checklist
- **Sentinel-2 / Landsat:** Open access via Copernicus/USGS (academic-friendly).
- **ERA5 / CHIRPS:** Open access via ECMWF/UCSB.
- **SoilGrids:** Open access via ISRIC.
- **SMAP / ESA CCI Soil Moisture:** Open access via NASA/ESA.
- **ESA WorldCereal Irrigation Maps:** Free for academic and open-source use.
- **ISRO / Po Valley Agricultural Registries:** Require registration or standard academic API requests.

---

## 6. Code Style Guidelines
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
      """
      def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
          super().__init__()
          self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
          self.norm = nn.LayerNorm(d_model)

      def forward(self, fine_tokens: torch.Tensor, coarse_tokens: torch.Tensor) -> torch.Tensor:
          # fine_tokens shape: [B, S_fine, D]
          # coarse_tokens shape: [B, S_coarse, D]
          attn_output, _ = self.cross_attn(query=fine_tokens, key=coarse_tokens, value=coarse_tokens)
          return self.norm(fine_tokens + attn_output)
  ```

---

## 7. Testing Strategy
- **Framework:** PyTest.
- **Coverage Target:** Minimum 85% overall coverage, 100% on physics loss formulas.
- **Unit Test Boundaries:**
  - `test_datacube.py` must verify that data loader masks are strictly zero-leak (no future time steps visible in past inputs).
  - `test_physics.py` must check that water-balance loss is 0.0 when parameters are perfectly balanced.
  - `test_diagnostics.py` must check the autocyclicity of the causal Directed Acyclic Graph (DAG) and verify gradient paths do not flow backward in time.

---

## 8. Boundaries
- **Always do:** Run tests and linter before commits; document every mathematical formula in code; follow PEP 8.
- **Ask first:** Modifying the Spatial-Temporal split years; adding new external data sources; changing physics loss terms.
- **Never do:** Resample coarse climate variables to 10m without scale-aware cross-attention; assume irrigation is fully observed without using $I_{\text{latent}}$ balance; use random pixel-level t-tests for EO statistical tests.

---

## 9. Success Criteria
- **Model Feasibility:** Multi-resolution dataloader parses Sentinel-2 (10m) and ERA5 (31km) into distinct tokens correctly.
- **Physics Success:** Weak water-balance regularization reduces physical water-budget RMSE by $\ge 15\%$ compared to unconstrained baseline, with $\le 3\%$ decrease in downstream metrics.
- **Crop Stress AUROC:** Downstream stress detection AUROC $\ge 0.90$ in-region, and $\ge 0.80$ in spatial/temporal transfer.
- **Phenology MAE:** Phenology stage transition prediction timing error $\le 5.0$ days.
- **Yield Risk $R^2$:** District yield anomaly forecast $R^2 \ge 0.70$ on in-region tests.
- **Causal Plausibility:** Temporal DAG constraints achieve 100% directional sign consistency with agronomic priors (e.g. higher temperature accelerates AGDD).
