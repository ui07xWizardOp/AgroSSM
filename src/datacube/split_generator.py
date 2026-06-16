import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

def assign_spatial_block(lat: float, lon: float, block_size_km: float = 50.0) -> Tuple[int, int]:
    """
    Assign a (lat, lon) coordinate to a spatial block on a grid of roughly block_size_km.
    At ~41 N (US Corn Belt), 1 degree lat is ~111 km, 1 degree lon is ~84 km.
    """
    lat_km = lat * 111.0
    # cos(41 degrees) is roughly 0.75
    lon_km = lon * 111.0 * 0.75
    
    lat_block = int(np.floor(lat_km / block_size_km))
    lon_block = int(np.floor(lon_km / block_size_km))
    return lat_block, lon_block

def generate_splits(
    metadata_list: List[Dict[str, Any]], 
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Generate spatiotemporal blocked cross-validation indices.
    
    Args:
        metadata_list: List of dictionaries containing:
            "patch_id" (str), "county_fips" (str), "year" (int), "lat" (float), "lon" (float)
        n_folds: Number of CV folds.
        
    Returns:
        A dictionary:
        {
            "folds": {
                0: {"train_indices": [...], "val_indices": [...]},
                ...
            },
            "test_indices": [...],       # 2022 patches
            "extreme_indices": [...]     # 2023 patches
        }
    """
    df = pd.DataFrame(metadata_list)
    
    # 1. Separate test and extreme holdouts (INV-D2, INV-D4)
    test_mask = df["year"] == 2022
    extreme_mask = df["year"] == 2023
    train_val_mask = df["year"].isin([2017, 2018, 2019, 2020, 2021])
    
    test_indices = df[test_mask].index.tolist()
    extreme_indices = df[extreme_mask].index.tolist()
    
    train_val_df = df[train_val_mask].copy()
    
    # 2. Assign spatial blocks
    blocks = []
    for _, row in train_val_df.iterrows():
        lat_b, lon_b = assign_spatial_block(row["lat"], row["lon"])
        blocks.append(f"{lat_b}_{lon_b}")
    train_val_df["block_id"] = blocks
    
    unique_blocks = sorted(train_val_df["block_id"].unique())
    n_blocks = len(unique_blocks)
    
    # Deterministic assignment of blocks to folds
    # Set seed to ensure reproducibility
    rng = np.random.default_rng(seed=42)
    shuffled_blocks = rng.permutation(unique_blocks)
    
    # Divide blocks into n_folds groups
    block_groups = np.array_split(shuffled_blocks, n_folds)
    
    folds = {}
    for fold_id in range(n_folds):
        val_blocks = set(block_groups[fold_id])
        
        # Train blocks are all blocks not in validation
        train_blocks = set(unique_blocks) - val_blocks
        
        # Map back to dataframe indices
        val_indices = train_val_df[train_val_df["block_id"].isin(val_blocks)].index.tolist()
        train_indices = train_val_df[train_val_df["block_id"].isin(train_blocks)].index.tolist()
        
        folds[fold_id] = {
            "train_indices": train_indices,
            "val_indices": val_indices
        }
        
    return {
        "folds": folds,
        "test_indices": test_indices,
        "extreme_indices": extreme_indices
    }
