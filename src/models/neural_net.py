import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Conv1DSeizureNet:
    """
    1D Convolutional Neural Network baseline for raw EEG window classification.
    Operates if PyTorch is available; gracefully handles environments where torch is missing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def _build_model(self, sequence_length: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use Conv1DSeizureNet.")

        torch.manual_seed(self.random_state)
        self.model = nn.Sequential(
            nn.Conv1d(self.in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_classes),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, class_weights: Optional[np.ndarray] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to train Conv1DSeizureNet.")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (N, 1, T)

        seq_len = X.shape[-1]
        self._build_model(seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model is not fitted or PyTorch is unavailable.")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

