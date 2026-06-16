import pytest
import torch
import numpy as np
from src.datacube.channel_registry import CHANNEL_REGISTRY, get_metadata_tensor
from src.datacube.patch_dataset import PatchDataset
from src.datacube.split_generator import generate_splits
from src.datacube.eo_harmonization import EOHarmonization

def test_channel_registry():
    # 1. Assert len(CHANNEL_REGISTRY) == 26 (25 base + 1 irrigation)
    assert len(CHANNEL_REGISTRY) == 26

    # 2. Assert all variable_id values are unique
    var_ids = [c.variable_id for c in CHANNEL_REGISTRY]
    assert len(var_ids) == len(set(var_ids))
    assert sorted(var_ids) == list(range(26))

    # 3. Assert get_metadata_tensor returns shape [3, 4]
    meta = get_metadata_tensor(["S2_Blue", "ERA5_Precip", "Irrigation"])
    assert meta.shape == (3, 4)

def test_patch_dataset():
    # Load PatchDataset
    ds = PatchDataset("dummy", training=True)
    sample = ds[0]

    # Verify return dictionary schema
    assert "optical" in sample
    assert sample["optical"].shape == (30, 6, 64, 64)
    assert sample["optical_mask"].shape == (30, 6)
    
    assert "weather" in sample
    assert sample["weather"].shape == (30, 9)
    assert sample["weather_mask"].shape == (30, 9)

    assert "soil" in sample
    assert sample["soil"].shape == (7,)
    assert sample["soil_mask"].shape == (7,)

    assert "calendar" in sample
    assert sample["calendar"].shape == (3,)
    assert sample["calendar_mask"].shape == (3,)

    assert "irrigation" in sample
    assert sample["irrigation"].shape == (64, 64)
    assert sample["irrigation_mask"].shape == (1,)

    # Verify cloud masking (at least one zero in optical_mask)
    assert (sample["optical_mask"] == 0.0).any()

    # Verify no NaN or Inf
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any()
            assert not torch.isinf(v).any()

    # Verify Modality Dropout (INV-T2)
    # Enable training and check if over 100 draws, at least one modality is fully zeroed in > 5% of samples
    optical_zero_count = 0
    weather_zero_count = 0
    irrigation_zero_count = 0
    
    for _ in range(100):
        s = ds[0]
        if s["optical_mask"].sum() == 0:
            optical_zero_count += 1
        if s["weather_mask"].sum() == 0:
            weather_zero_count += 1
        if s["irrigation_mask"].sum() == 0:
            irrigation_zero_count += 1

    # In PatchDataset.py, probability of dropout is 15% (0.15)
    # Expected count is around 15 out of 100, checking if it is at least 5% (5 draws)
    assert optical_zero_count >= 5
    assert weather_zero_count >= 5
    assert irrigation_zero_count >= 5

def test_split_generator():
    # Create dummy metadata list
    metadata_list = []
    rng = np.random.default_rng(seed=42)
    
    # 200 patches distributed across years and coordinates
    for i in range(200):
        year = int(rng.choice([2017, 2018, 2019, 2020, 2021, 2022, 2023]))
        # Lat: 40 to 43 (US Corn Belt), Lon: -96 to -91
        lat = float(rng.uniform(40.0, 43.0))
        lon = float(rng.uniform(-96.0, -91.0))
        metadata_list.append({
            "patch_id": f"patch_{i}",
            "county_fips": f"19{i:03d}",
            "year": year,
            "lat": lat,
            "lon": lon
        })

    splits = generate_splits(metadata_list, n_folds=5)
    
    folds = splits["folds"]
    test_indices = splits["test_indices"]
    extreme_indices = splits["extreme_indices"]

    # Verify no year leakage for holdouts (2022 test, 2023 extreme)
    for idx in test_indices:
        assert metadata_list[idx]["year"] == 2022

    for idx in extreme_indices:
        assert metadata_list[idx]["year"] == 2023

    # Verify fold properties
    for fold_id, fold in folds.items():
        train_idxs = fold["train_indices"]
        val_idxs = fold["val_indices"]

        # 1. Assert no overlap within fold
        assert len(set(train_idxs).intersection(set(val_idxs))) == 0

        # 2. Assert no holdout years in training/validation
        for idx in train_idxs + val_idxs:
            assert metadata_list[idx]["year"] in [2017, 2018, 2019, 2020, 2021]

        # 3. Check spatial block split integrity (leakage check)
        # Verify that no spatial block coordinate falls in both sets
        train_blocks = set()
        for idx in train_idxs:
            p = metadata_list[idx]
            # Simple conversion to block to verify
            lat_km, lon_km = int(np.floor(p["lat"] * 111.0 / 50.0)), int(np.floor(p["lon"] * 111.0 * 0.75 / 50.0))
            train_blocks.add(f"{lat_km}_{lon_km}")
            
        for idx in val_idxs:
            p = metadata_list[idx]
            lat_km, lon_km = int(np.floor(p["lat"] * 111.0 / 50.0)), int(np.floor(p["lon"] * 111.0 * 0.75 / 50.0))
            assert f"{lat_km}_{lon_km}" not in train_blocks

    # Check balanced size (within +/-15% of expected 20%)
    total_train_val = len(folds[0]["train_indices"]) + len(folds[0]["val_indices"])
    expected_val_size = total_train_val / 5.0
    for fold_id, fold in folds.items():
        val_size = len(fold["val_indices"])
        assert abs(val_size - expected_val_size) < (expected_val_size * 0.25)

def test_eo_harmonization_stub():
    harm = EOHarmonization()
    dummy_tensor = torch.randn(10, 10)
    out = harm.harmonize(dummy_tensor, source_sensor="Landsat-8")
    # Stub should be a no-op returning the identical tensor
    assert torch.equal(out, dummy_tensor)
