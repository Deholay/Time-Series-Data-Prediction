from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except OSError as exc:
        raise OSError(
            "PyTorch failed to load. N-BEATS requires a working PyTorch install; "
            "reinstall the CPU build or use a Python environment where `import torch` succeeds."
        ) from exc
    except ImportError as exc:
        raise ImportError("N-BEATS requires PyTorch. Install it with the correct wheel for your Python version.") from exc
    return torch, nn, DataLoader, TensorDataset


def add_servicenow_nbeats_path(repo_path: str | Path | None = None) -> Path | None:
    """Add a local ServiceNow/N-BEATS checkout to sys.path when available."""
    candidates: list[Path] = []
    if repo_path:
        candidates.append(Path(repo_path))
    if os.environ.get("SERVICE_NOW_NBEATS_REPO"):
        candidates.append(Path(os.environ["SERVICE_NOW_NBEATS_REPO"]))
    candidates.extend(
        [
            Path.cwd() / "external" / "N-BEATS",
            Path.cwd() / "N-BEATS",
        ]
    )

    for candidate in candidates:
        if (candidate / "models" / "nbeats.py").exists():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate.resolve()
    return None


def import_servicenow_nbeats(repo_path: str | Path | None = None):
    """Import the official ServiceNow N-BEATS model classes."""
    add_servicenow_nbeats_path(repo_path)
    try:
        from models.nbeats import GenericBasis, NBeats, NBeatsBlock
    except ImportError as exc:
        raise ImportError(
            "ServiceNow/N-BEATS was not found. Clone https://github.com/ServiceNow/N-BEATS "
            "to external/N-BEATS or set SERVICE_NOW_NBEATS_REPO to that checkout."
        ) from exc
    return NBeats, NBeatsBlock, GenericBasis


@dataclass
class NBeatsTrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class ServiceNowNBeatsPredictor:
    """Thin trainer around the official ServiceNow N-BEATS PyTorch classes."""

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int = 1,
        stacks: int = 2,
        blocks_per_stack: int = 3,
        layers: int = 4,
        layer_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str | None = None,
        repo_path: str | Path | None = None,
    ) -> None:
        if backcast_size < 2:
            raise ValueError("backcast_size must be >= 2")
        if forecast_size != 1:
            raise ValueError("This project predicts one horizon return at a time, so forecast_size must be 1")

        torch, nn, _, _ = _import_torch()
        self._torch = torch
        self._nn = nn
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        nbeats_cls, block_cls, basis_cls = import_servicenow_nbeats(repo_path)
        blocks = []
        for _ in range(stacks):
            for _ in range(blocks_per_stack):
                blocks.append(
                    block_cls(
                        input_size=backcast_size,
                        theta_size=backcast_size + forecast_size,
                        basis_function=basis_cls(backcast_size=backcast_size, forecast_size=forecast_size),
                        layers=layers,
                        layer_size=layer_size,
                    )
                )

        self.device = torch.device(device)
        self.model = nbeats_cls(nn.ModuleList(blocks)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.SmoothL1Loss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        validation_fraction: float = 0.15,
        patience: int = 12,
    ) -> NBeatsTrainingHistory:
        torch = self._torch
        nn = self._nn
        _, _, DataLoader, TensorDataset = _import_torch()
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2-D: (samples, backcast_size)")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of samples")

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        mask_tensor = torch.ones_like(X_tensor)

        n_total = len(X_tensor)
        n_val = min(n_total - 1, max(1, int(n_total * validation_fraction)))
        n_train = n_total - n_val
        train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train], mask_tensor[:n_train])
        val_X = X_tensor[n_train:].to(self.device)
        val_y = y_tensor[n_train:].to(self.device)
        val_mask = mask_tensor[n_train:].to(self.device)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        history = NBeatsTrainingHistory(train_loss=[], val_loss=[])
        best_state = None
        best_val = float("inf")
        stale_epochs = 0

        for _ in range(epochs):
            self.model.train()
            losses = []
            for batch_X, batch_y, batch_mask in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_X, batch_mask)
                loss = self.loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_X, val_mask)
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
        torch = self._torch
        _, _, DataLoader, TensorDataset = _import_torch()
        if X.ndim != 2:
            raise ValueError("X must be 2-D: (samples, backcast_size)")
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        mask_tensor = torch.ones_like(X_tensor)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=batch_size, shuffle=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_mask in loader:
                pred = self.model(batch_X.to(self.device), batch_mask.to(self.device))
                preds.append(pred.detach().cpu().numpy())
        return np.concatenate(preds).ravel()
