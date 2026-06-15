from __future__ import annotations

import numpy as np
import pandas as pd


INDEX_TICKERS = {
    "Nasdaq": "^IXIC",
    "SP500": "^GSPC",
    "SOX": "^SOX",
}


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, naive: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    naive = np.asarray(naive)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    mape = float(np.mean(np.abs((predicted - actual) / actual)) * 100.0)
    naive_rmse = float(np.sqrt(np.mean((naive - actual) ** 2)))
    naive_mae = float(np.mean(np.abs(naive - actual)))
    return {
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "naive_rmse": naive_rmse,
        "naive_mae": naive_mae,
        "rmse_vs_naive_pct": float((1.0 - rmse / naive_rmse) * 100.0) if naive_rmse else np.nan,
        "mae_vs_naive_pct": float((1.0 - mae / naive_mae) * 100.0) if naive_mae else np.nan,
    }


def direction_accuracy(actual: pd.Series, predicted: pd.Series, previous_close: pd.Series) -> float:
    actual_direction = np.sign(actual.to_numpy() - previous_close.to_numpy())
    predicted_direction = np.sign(predicted.to_numpy() - previous_close.to_numpy())
    return float((actual_direction == predicted_direction).mean() * 100.0)
