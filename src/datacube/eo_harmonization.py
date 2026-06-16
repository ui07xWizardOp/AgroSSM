import logging
import torch

logger = logging.getLogger(__name__)

class EOHarmonization:
    """
    Stub for EO (Earth Observation) Harmonization.
    
    This class documents the scaling path for Sensor-Mix OOD (Landsat/Sentinel-2 harmonization).
    Since the pilot scope is limited to Sentinel-2 only, this class acts as a no-op.
    
    See Section 13 (Scaling Projection) of the Architecture Spec for the full integration path.
    """
    def __init__(self, target_sensor: str = "Sentinel-2"):
        self.target_sensor = target_sensor
        logger.info("Initializing EO Harmonization stub (no-op for Sentinel-only pilot)")

    def harmonize(self, data: torch.Tensor, source_sensor: str) -> torch.Tensor:
        """
        No-op harmonization function returning the input data unmodified.
        """
        if source_sensor != self.target_sensor:
            # Under a full foundation model, this would apply a spectral band adjustment factor (SBAF)
            # and spatial resampling to align Landsat and Sentinel-2 sensors.
            pass
        return data
