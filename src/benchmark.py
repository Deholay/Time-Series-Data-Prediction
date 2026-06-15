from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from indicator import add_technical_indicators, select_features_by_ic
from LSTM import LSTMPredictor
from tensor_transform import build_lstm_tensors


INDEX_TICKERS = {
    "Nasdaq": "^IXIC",
    "SP500": "^GSPC",
    "SOX": "^SOX",
}


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


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


def run_benchmark(
    name: str,
    ticker: str,
    start: str,
    end: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lookback: int,
    horizon: int,
    top_k: int,
    epochs: int,
    output_dir: Path,
) -> dict:
    raw = download_index(ticker, start=start, end=end)
    features_df = add_technical_indicators(raw)

    train_features_df = features_df.loc[:train_end]
    selection = select_features_by_ic(train_features_df, horizon=horizon, top_k=top_k)
    tensor_data = build_lstm_tensors(
        features_df,
        feature_columns=selection.features,
        lookback=lookback,
        horizon=horizon,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        target_mode="log_return",
    )

    model = LSTMPredictor(input_size=len(selection.features))
    history = model.fit(tensor_data.X_train, tensor_data.y_train, epochs=epochs)

    pred_scaled = model.predict(tensor_data.X_test)
    predicted_return = tensor_data.inverse_target(pred_scaled)
    actual_return = tensor_data.inverse_target(tensor_data.y_test)
    predicted = tensor_data.test_base_values * np.exp(predicted_return)
    actual = tensor_data.test_base_values * np.exp(actual_return)
    test_dates = tensor_data.test_dates

    baseline = tensor_data.test_base_values

    prediction_df = pd.DataFrame(
        {
            "forecast_base_date": tensor_data.test_base_dates,
            "date": test_dates,
            "index": name,
            "ticker": ticker,
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
        pd.Series(actual, index=test_dates),
        pd.Series(predicted, index=test_dates),
        pd.Series(baseline, index=test_dates),
    )
    metrics["samples"] = int(len(actual))
    metrics["train_samples"] = int(len(tensor_data.y_train))
    metrics["features"] = selection.features
    metrics["final_train_loss"] = history.train_loss[-1]
    metrics["final_val_loss"] = history.val_loss[-1]

    selection.scores.to_csv(output_dir / f"{name}_feature_ic.csv", index=False)
    raw.to_csv(output_dir / f"{name}_raw.csv")
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM through Dec 2025 and benchmark Jan-May 2026 forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running {name} ({ticker})...")
        result = run_benchmark(
            name=name,
            ticker=ticker,
            start=args.start,
            end=args.end,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            lookback=args.lookback,
            horizon=args.horizon,
            top_k=args.top_k,
            epochs=args.epochs,
            output_dir=output_dir,
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
