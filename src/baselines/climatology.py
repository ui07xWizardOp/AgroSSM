import numpy as np
from typing import Optional

class ClimatologyBaseline:
    """Historical mean prediction baseline."""
    def __init__(self):
        self.mean_val = 0.0

    def fit(self, X_train: Optional[np.ndarray], y_train: np.ndarray):
        self.mean_val = np.mean(y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(X_test.shape[0], self.mean_val)
