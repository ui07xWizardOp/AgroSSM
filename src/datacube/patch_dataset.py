import torch
from torch.utils.data import Dataset
from typing import Optional, Dict

class PatchDataset(Dataset):
    """
    PyTorch Dataset class that loads one patch's multi-resolution time series
    and returns properly shaped tensors.
    """
    def __init__(self, data_path: str, training: bool = True):
        super().__init__()
        self.data_path = data_path
        self.training = training
        # Placeholder for actual data loading logic
        self.num_samples = 100

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads one patch, applies normalization using training-set statistics (INV-D3),
        applies modality dropout if self.training (INV-T2), and returns the dictionary.
        """
        # Dummy data matching the schema required in documentation
        # In a real scenario, this would load data from .npz or .zarr
        T = 30
        C_opt = 6
        C_weather = 9
        C_soil = 7
        C_cal = 3
        H, W = 64, 64

        # Base modalities
        optical = torch.randn(T, C_opt, H, W)
        weather = torch.randn(T, C_weather)
        soil = torch.randn(C_soil)
        calendar = torch.randn(C_cal)
        irrigation = torch.randint(0, 2, (H, W)).float()

        # Masks
        optical_mask = torch.ones(T, C_opt)
        # Introduce some fake cloud masking for testing
        optical_mask[5, :] = 0.0

        weather_mask = torch.ones(T, C_weather)
        soil_mask = torch.ones(C_soil)
        calendar_mask = torch.ones(C_cal)
        irrigation_mask = torch.tensor([1.0])

        # Modality dropout (INV-T2)
        if self.training:
            if torch.rand(1).item() < 0.15: # 15% chance to drop optical
                optical.zero_()
                optical_mask.zero_()
            if torch.rand(1).item() < 0.15:
                weather.zero_()
                weather_mask.zero_()
            if torch.rand(1).item() < 0.15:
                irrigation.zero_()
                irrigation_mask.zero_()

        return {
            "optical": optical,
            "optical_mask": optical_mask,
            "weather": weather,
            "weather_mask": weather_mask,
            "soil": soil,
            "soil_mask": soil_mask,
            "calendar": calendar,
            "calendar_mask": calendar_mask,
            "irrigation": irrigation,
            "irrigation_mask": irrigation_mask,
            "patch_id": f"patch_{idx}",
            "county_fips": "19001",
            "year": 2020,
            "doy_sequence": torch.arange(91, 91 + T * 5, 5).float(),
            "lat": 41.5,
            "lon": -93.5,
            "stress_label": torch.randint(0, 2, (T, 4, 4)).float(),
            "phenology_label": torch.randn(T),
            "yield_anomaly": 5.0
        }
