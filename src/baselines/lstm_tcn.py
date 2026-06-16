import torch
import torch.nn as nn
import numpy as np

class LSTMBaselineModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.out(h_n[-1])

class LSTMBaseline:
    """LSTM baseline on time-series features."""
    def __init__(self, input_dim: int):
        self.model = LSTMBaselineModel(input_dim)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X_test, dtype=torch.float32)
            return self.model(x).squeeze().numpy()
