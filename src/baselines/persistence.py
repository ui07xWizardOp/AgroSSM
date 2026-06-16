import numpy as np

class PersistenceBaseline:
    """Predicts the last observed value."""
    def __init__(self):
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Assuming last feature or specific index represents the last observed value
        # Here we just take the last column as a dummy implementation
        return X_test[:, -1]
