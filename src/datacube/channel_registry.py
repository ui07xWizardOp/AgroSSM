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
import torch

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
