from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

from benchmark_utils import INDEX_TICKERS, direction_accuracy, regression_metrics
from indicator import add_technical_indicators, select_features_by_ic
from tensor_transform import build_lstm_tensors


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.Chomp1d(padding) if hasattr(nn, "Chomp1d") else Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.Chomp1d(padding) if hasattr(nn, "Chomp1d") else Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size] if self.chomp_size else x


class BiLSTMMTRANTCN(nn.Module):
    """BiLSTM -> modified Transformer encoder -> dilated TCN decoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        model_dim = hidden_size * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.mtran = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.tcn = nn.Sequential(
            TemporalBlock(model_dim, dilation=1, dropout=dropout),
            TemporalBlock(model_dim, dilation=2, dropout=dropout),
            TemporalBlock(model_dim, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x)
        x, _ = self.bilstm(x)
        x = self.mtran(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x[:, -1, :]).squeeze(-1)


class HybridPredictor:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = BiLSTMMTRANTCN(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.SmoothL1Loss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = 32) -> TrainingHistory:
        X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        n_val = min(len(X_tensor) - 1, max(1, int(len(X_tensor) * 0.15)))
        n_train = len(X_tensor) - n_val
        loader = DataLoader(TensorDataset(X_tensor[:n_train], y_tensor[:n_train]), batch_size=batch_size, shuffle=True)
        val_X = X_tensor[n_train:].to(self.device)
        val_y = y_tensor[n_train:].to(self.device)
        history = TrainingHistory(train_loss=[], val_loss=[])
        best_state = None
        best_val = float("inf")
        stale = 0
        for _ in range(epochs):
            self.model.train()
            losses = []
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model(batch_X), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                val_loss = float(self.loss_fn(self.model(val_X), val_y).detach().cpu())
            history.train_loss.append(float(np.mean(losses)))
            history.val_loss.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 12:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (batch_X,) in loader:
                preds.append(self.model(batch_X.to(self.device)).detach().cpu().numpy())
        return np.concatenate(preds)


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def run_hybrid_benchmark(
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
    selection = select_features_by_ic(features_df.loc[:train_end], horizon=horizon, top_k=top_k)
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
    model = HybridPredictor(input_size=len(selection.features))
    history = model.fit(tensor_data.X_train, tensor_data.y_train, epochs=epochs)
    pred_scaled = model.predict(tensor_data.X_test)
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
            "model": "BiLSTM-MTRAN-TCN",
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
    selection.scores.to_csv(output_dir / f"{name}_feature_ic.csv", index=False)
    raw.to_csv(output_dir / f"{name}_raw.csv")
    metrics = regression_metrics(actual, predicted, baseline)
    metrics["direction_accuracy_pct"] = direction_accuracy(
        pd.Series(actual, index=tensor_data.test_dates),
        pd.Series(predicted, index=tensor_data.test_dates),
        pd.Series(baseline, index=tensor_data.test_dates),
    )
    metrics["samples"] = int(len(actual))
    metrics["train_samples"] = int(len(tensor_data.y_train))
    metrics["model"] = "BiLSTM-MTRAN-TCN"
    metrics["features"] = selection.features
    metrics["final_train_loss"] = history.train_loss[-1]
    metrics["final_val_loss"] = history.val_loss[-1]
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTM-MTRAN-TCN benchmark forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", default="outputs_bilstm_mtran_tcn_h21")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running BiLSTM-MTRAN-TCN {name} ({ticker})...")
        result = run_hybrid_benchmark(
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
        print(f"{name}: RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}, MAPE={result['mape_pct']:.2f}%")
    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
