from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from benchmark_utils import INDEX_TICKERS, direction_accuracy, regression_metrics
from indicator import add_technical_indicators, select_features_by_ic
from tensor_transform import build_lstm_tensors


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install Microsoft LightGBM first: pip install lightgbm") from exc
    return lgb


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def flatten_lag_features(X: np.ndarray, feature_names: list[str], lookback: int) -> tuple[np.ndarray, list[str]]:
    if X.ndim != 3:
        raise ValueError("X must be 3-D: (samples, lookback, features)")
    flat = X.reshape(X.shape[0], -1)
    names = []
    for lag in range(lookback, 0, -1):
        for feature in feature_names:
            names.append(f"{feature}_lag_{lag}")
    return flat, names


def run_lightgbm_benchmark(
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
    output_dir: Path,
    n_estimators: int = 600,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
) -> dict:
    lgb = _import_lightgbm()
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

    X_train, flat_feature_names = flatten_lag_features(tensor_data.X_train, selection.features, lookback)
    X_test, _ = flatten_lag_features(tensor_data.X_test, selection.features, lookback)
    y_train = tensor_data.y_train

    val_size = min(len(X_train) - 1, max(1, int(len(X_train) * 0.15)))
    train_size = len(X_train) - val_size
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train[:train_size],
        y_train[:train_size],
        eval_set=[(X_train[train_size:], y_train[train_size:])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred_scaled = model.predict(X_test, num_iteration=model.best_iteration_)
    predicted_return = tensor_data.inverse_target(pred_scaled)
    actual_return = tensor_data.inverse_target(tensor_data.y_test)
    predicted = tensor_data.test_base_values * np.exp(predicted_return)
    actual = tensor_data.test_base_values * np.exp(actual_return)
    baseline = tensor_data.test_base_values

    prediction_df = pd.DataFrame(
        {
            "forecast_base_date": tensor_data.test_base_dates,
            "date": tensor_data.test_dates,
            "index": name,
            "ticker": ticker,
            "model": "LightGBM",
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
        pd.Series(actual, index=tensor_data.test_dates),
        pd.Series(predicted, index=tensor_data.test_dates),
        pd.Series(baseline, index=tensor_data.test_dates),
    )
    metrics["samples"] = int(len(actual))
    metrics["train_samples"] = int(len(tensor_data.y_train))
    metrics["model"] = "LightGBM"
    metrics["features"] = selection.features
    train_pred = model.predict(X_train[:train_size], num_iteration=model.best_iteration_)
    val_pred = model.predict(X_train[train_size:], num_iteration=model.best_iteration_)
    metrics["final_train_loss"] = float(np.mean((train_pred - y_train[:train_size]) ** 2))
    metrics["final_val_loss"] = float(np.mean((val_pred - y_train[train_size:]) ** 2))
    metrics["best_iteration"] = int(model.best_iteration_ or n_estimators)

    selection.scores.to_csv(output_dir / f"{name}_feature_ic.csv", index=False)
    pd.DataFrame(
        {
            "feature": flat_feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(output_dir / f"{name}_feature_importance.csv", index=False)
    raw.to_csv(output_dir / f"{name}_raw.csv")
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Microsoft LightGBM benchmark forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs_lightgbm_h21")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running LightGBM {name} ({ticker})...")
        result = run_lightgbm_benchmark(
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
            output_dir=output_dir,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
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
