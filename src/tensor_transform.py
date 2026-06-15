from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TensorData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    train_base_dates: pd.DatetimeIndex
    test_base_dates: pd.DatetimeIndex
    train_base_values: np.ndarray
    test_base_values: np.ndarray
    feature_columns: list[str]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    target_mode: str

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        values_2d = np.asarray(values).reshape(-1, 1)
        return self.target_scaler.inverse_transform(values_2d).ravel()


def _validate_dates(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    if dates.hasnans:
        raise ValueError("DataFrame index must be date-like and cannot contain NaT")
    return pd.DatetimeIndex(dates)


def build_lstm_tensors(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "Close",
    lookback: int = 30,
    horizon: int = 1,
    train_end: str | pd.Timestamp = "2025-12-31",
    test_start: str | pd.Timestamp = "2026-01-01",
    test_end: str | pd.Timestamp = "2026-05-31",
    target_mode: str = "log_return",
) -> TensorData:
    """Create 3-D LSTM tensors: (batch, time_steps, input_features)."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if target_mode not in {"log_return", "close"}:
        raise ValueError("target_mode must be 'log_return' or 'close'")

    missing = [c for c in list(feature_columns) + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df.copy().sort_index()
    data.index = _validate_dates(data.index)
    feature_columns = list(feature_columns)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    needed = feature_columns + [target_col]
    data = data[needed].replace([np.inf, -np.inf], np.nan).dropna()

    X_raw, y_raw, base_raw, base_dates, sample_dates = [], [], [], [], []
    values = data[feature_columns].to_numpy(dtype=np.float32)
    target = data[target_col].to_numpy(dtype=np.float32)

    for end_idx in range(lookback - 1, len(data) - horizon):
        target_idx = end_idx + horizon
        pred_date = data.index[target_idx]
        base_value = target[end_idx]
        target_value = target[target_idx]
        if target_mode == "log_return":
            y_value = np.log(target_value / base_value)
        else:
            y_value = target_value
        X_raw.append(values[end_idx - lookback + 1 : end_idx + 1])
        y_raw.append(y_value)
        base_raw.append(base_value)
        base_dates.append(data.index[end_idx])
        sample_dates.append(pred_date)

    if not X_raw:
        raise ValueError("Not enough rows to build any sequences")

    X_raw = np.asarray(X_raw, dtype=np.float32)
    y_raw = np.asarray(y_raw, dtype=np.float32).reshape(-1, 1)
    base_raw = np.asarray(base_raw, dtype=np.float32)
    base_dates = pd.DatetimeIndex(base_dates)
    sample_dates = pd.DatetimeIndex(sample_dates)

    train_mask = sample_dates <= train_end
    test_mask = (sample_dates >= test_start) & (sample_dates <= test_end)
    if train_mask.sum() == 0:
        raise ValueError("No training samples before train_end")
    if test_mask.sum() == 0:
        raise ValueError("No test samples in requested prediction range")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    n_features = X_raw.shape[-1]

    feature_scaler.fit(X_raw[train_mask].reshape(-1, n_features))
    target_scaler.fit(y_raw[train_mask])

    X_scaled = feature_scaler.transform(X_raw.reshape(-1, n_features)).reshape(X_raw.shape)
    y_scaled = target_scaler.transform(y_raw).ravel()

    return TensorData(
        X_train=X_scaled[train_mask].astype(np.float32),
        y_train=y_scaled[train_mask].astype(np.float32),
        X_test=X_scaled[test_mask].astype(np.float32),
        y_test=y_scaled[test_mask].astype(np.float32),
        train_dates=sample_dates[train_mask],
        test_dates=sample_dates[test_mask],
        train_base_dates=base_dates[train_mask],
        test_base_dates=base_dates[test_mask],
        train_base_values=base_raw[train_mask],
        test_base_values=base_raw[test_mask],
        feature_columns=feature_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_mode=target_mode,
    )
