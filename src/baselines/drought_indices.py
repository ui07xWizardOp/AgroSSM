import numpy as np

class DroughtIndicesBaseline:
    """SPI/SPEI/VHI based baseline."""
    def __init__(self):
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.zeros(X_test.shape[0])
