from sklearn.ensemble import RandomForestRegressor
import numpy as np

class TreeBaseline:
    """Random Forest / XGBoost baseline on handcrafted features."""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
