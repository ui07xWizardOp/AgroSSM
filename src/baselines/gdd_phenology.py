import numpy as np

class GDDPhenologyBaseline:
    """GDD thermal-time model for phenology."""
    def __init__(self, t_base: float = 10.0):
        self.t_base = t_base
        self.threshold = None

    def fit(self, X_train_temp: np.ndarray, y_train_days: np.ndarray):
        # Dummy fit to find an average GDD threshold
        self.threshold = 500.0

    def predict(self, X_test_temp: np.ndarray) -> np.ndarray:
        # Dummy prediction
        return np.full(X_test_temp.shape[0], 10.0)
