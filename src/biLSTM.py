from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PriceBiLSTM(nn.Module):
    """Bidirectional LSTM forecaster for 3-D market tensors."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.bilstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        bidirectional_state = torch.cat((forward_hidden, backward_hidden), dim=1)
        return self.head(bidirectional_state).squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class BiLSTMPredictor:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = PriceBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        validation_fraction: float = 0.15,
        patience: int = 12,
    ) -> TrainingHistory:
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of samples")
        if len(X_train) < 2:
            raise ValueError("At least two training samples are required")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32)

        n_total = len(X_tensor)
        n_val = min(n_total - 1, max(1, int(n_total * validation_fraction)))
        n_train = n_total - n_val
        train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        val_X = X_tensor[n_train:].to(self.device)
        val_y = y_tensor[n_train:].to(self.device)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        history = TrainingHistory(train_loss=[], val_loss=[])
        best_state = None
        best_val = float("inf")
        stale_epochs = 0

        for _ in range(epochs):
            self.model.train()
            losses = []
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_X)
                loss = self.loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_X)
                val_loss = float(self.loss_fn(val_pred, val_y).detach().cpu())

            history.train_loss.append(float(np.mean(losses)))
            history.val_loss.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.model.eval()
        preds = []
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (batch_X,) in loader:
                pred = self.model(batch_X.to(self.device))
                preds.append(pred.detach().cpu().numpy())
        return np.concatenate(preds)


LSTMPredictor = BiLSTMPredictor

