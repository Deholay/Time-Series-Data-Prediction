from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from benchmark_utils import INDEX_TICKERS, direction_accuracy, regression_metrics
from indicator import normalize_ohlcv_columns
from nbeats_external import ServiceNowNBeatsPredictor, add_servicenow_nbeats_path


@dataclass
class NBeatsSequenceData:
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
    feature_scaler: StandardScaler
    target_scaler: StandardScaler

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    return normalize_ohlcv_columns(df)


def build_nbeats_sequences(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
) -> NBeatsSequenceData:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    data = normalize_ohlcv_columns(df).copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))
    data = data.sort_index()
    data["LogReturn_1d"] = np.log(data["Close"].astype(float) / data["Close"].astype(float).shift(1))
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close", "LogReturn_1d"])

    returns = data["LogReturn_1d"].to_numpy(dtype=np.float32)
    close = data["Close"].to_numpy(dtype=np.float32)

    X_raw, y_raw, base_raw, base_dates, sample_dates = [], [], [], [], []
    for end_idx in range(lookback - 1, len(data) - horizon):
        target_idx = end_idx + horizon
        base_value = close[end_idx]
        target_value = close[target_idx]
        X_raw.append(returns[end_idx - lookback + 1 : end_idx + 1])
        y_raw.append(np.log(target_value / base_value))
        base_raw.append(base_value)
        base_dates.append(data.index[end_idx])
        sample_dates.append(data.index[target_idx])

    if not X_raw:
        raise ValueError("Not enough rows to build N-BEATS sequences")

    X_raw = np.asarray(X_raw, dtype=np.float32)
    y_raw = np.asarray(y_raw, dtype=np.float32).reshape(-1, 1)
    base_raw = np.asarray(base_raw, dtype=np.float32)
    base_dates = pd.DatetimeIndex(base_dates)
    sample_dates = pd.DatetimeIndex(sample_dates)

    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    train_mask = sample_dates <= train_end
    test_mask = (sample_dates >= test_start) & (sample_dates <= test_end)
    if train_mask.sum() == 0:
        raise ValueError("No training samples before train_end")
    if test_mask.sum() == 0:
        raise ValueError("No test samples in requested prediction range")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(X_raw[train_mask])
    target_scaler.fit(y_raw[train_mask])
    X_scaled = feature_scaler.transform(X_raw)
    y_scaled = target_scaler.transform(y_raw).ravel()

    return NBeatsSequenceData(
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
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def run_nbeats_benchmark(
    name: str,
    ticker: str,
    start: str,
    end: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lookback: int,
    horizon: int,
    epochs: int,
    output_dir: Path,
    repo_path: str | Path | None = None,
) -> dict:
    raw = download_index(ticker, start=start, end=end)
    sequence_data = build_nbeats_sequences(
        raw,
        lookback=lookback,
        horizon=horizon,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    model = ServiceNowNBeatsPredictor(backcast_size=lookback, forecast_size=1, repo_path=repo_path)
    history = model.fit(sequence_data.X_train, sequence_data.y_train, epochs=epochs)
    pred_scaled = model.predict(sequence_data.X_test)

    predicted_return = sequence_data.inverse_target(pred_scaled)
    actual_return = sequence_data.inverse_target(sequence_data.y_test)
    predicted = sequence_data.test_base_values * np.exp(predicted_return)
    actual = sequence_data.test_base_values * np.exp(actual_return)
    baseline = sequence_data.test_base_values

    prediction_df = pd.DataFrame(
        {
            "forecast_base_date": sequence_data.test_base_dates,
            "date": sequence_data.test_dates,
            "index": name,
            "ticker": ticker,
            "model": "N-BEATS",
            "horizon_trading_days": horizon,
            "actual_close": actual,
            "predicted_close": predicted,
            "naive_previous_close": baseline,
            "actual_log_return": actual_return,
            "predicted_log_return": predicted_return,
            "absolute_error": np.abs(predicted - actual),
            "pct_error": np.abs((predicted - actual) / actual) * 100.0,
        }
    )
    prediction_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    metrics = regression_metrics(actual, predicted, baseline)
    metrics["direction_accuracy_pct"] = direction_accuracy(
        pd.Series(actual, index=sequence_data.test_dates),
        pd.Series(predicted, index=sequence_data.test_dates),
        pd.Series(baseline, index=sequence_data.test_dates),
    )
    metrics["samples"] = int(len(actual))
    metrics["train_samples"] = int(len(sequence_data.y_train))
    metrics["model"] = "N-BEATS"
    metrics["features"] = ["Close_LogReturn_1d"]
    metrics["final_train_loss"] = history.train_loss[-1]
    metrics["final_val_loss"] = history.val_loss[-1]

    raw.to_csv(output_dir / f"{name}_raw.csv")
    pd.DataFrame([{"feature": "Close_LogReturn_1d", "ic": np.nan, "abs_ic": np.nan}]).to_csv(
        output_dir / f"{name}_feature_ic.csv", index=False
    )
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ServiceNow N-BEATS benchmark forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", default="outputs_nbeats_h21")
    parser.add_argument("--nbeats-repo", default=None)
    args = parser.parse_args()

    add_servicenow_nbeats_path(args.nbeats_repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running N-BEATS {name} ({ticker})...")
        result = run_nbeats_benchmark(
            name=name,
            ticker=ticker,
            start=args.start,
            end=args.end,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            lookback=args.lookback,
            horizon=args.horizon,
            epochs=args.epochs,
            output_dir=output_dir,
            repo_path=args.nbeats_repo,
        )
        results.append(result)
        print(
            f"{name}: RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}, "
            f"MAPE={result['mape_pct']:.2f}%, Direction={result['direction_accuracy_pct']:.2f}%"
        )

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
