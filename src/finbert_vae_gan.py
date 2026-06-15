from __future__ import annotations

import argparse
import json
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
from lightgbm_benchmark import flatten_lag_features
from tensor_transform import build_lstm_tensors


@dataclass
class TrainingHistory:
    vae_loss: list[float]
    predictor_loss: list[float]
    val_loss: list[float]


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def load_finbert_sentiment(
    news_csv: str | Path | None,
    dates: pd.DatetimeIndex,
    model_name: str = "ProsusAI/finbert",
) -> pd.DataFrame:
    """Return per-date sentiment features. Without news data, return neutral features."""
    index = pd.DatetimeIndex(pd.to_datetime(dates).normalize()).unique().sort_values()
    neutral = pd.DataFrame(
        {"sentiment_positive": 0.0, "sentiment_negative": 0.0, "sentiment_neutral": 1.0},
        index=index,
    )
    if not news_csv:
        return neutral
    news = pd.read_csv(news_csv)
    if not {"date", "text"}.issubset(news.columns):
        raise ValueError("news_csv must contain date and text columns")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    if news.empty:
        return neutral

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("FinBERT sentiment requires transformers. Install requirements.txt first.") from exc

    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
    rows = []
    for date, group in news.groupby("date"):
        scores = {"positive": [], "negative": [], "neutral": []}
        for text in group["text"].dropna().astype(str).tolist():
            result = classifier(text[:512])[0]
            for item in result:
                scores[item["label"].lower()].append(float(item["score"]))
        rows.append(
            {
                "date": date,
                "sentiment_positive": float(np.mean(scores["positive"])) if scores["positive"] else 0.0,
                "sentiment_negative": float(np.mean(scores["negative"])) if scores["negative"] else 0.0,
                "sentiment_neutral": float(np.mean(scores["neutral"])) if scores["neutral"] else 1.0,
            }
        )
    sentiment = pd.DataFrame(rows).set_index("date")
    return sentiment.combine_first(neutral).fillna(
        {"sentiment_positive": 0.0, "sentiment_negative": 0.0, "sentiment_neutral": 1.0}
    )


class FeatureVAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int = 32) -> None:
        super().__init__()
        hidden = max(64, input_size // 2)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_size)
        self.logvar = nn.Linear(hidden, latent_size)
        self.decoder = nn.Sequential(nn.Linear(latent_size, hidden), nn.ReLU(), nn.Linear(hidden, input_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.decoder(z)
        return recon, mu, logvar, z


class LatentGenerator(nn.Module):
    def __init__(self, noise_size: int, latent_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_size, 64), nn.ReLU(), nn.Linear(64, latent_size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_size, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class PricePredictor(nn.Module):
    def __init__(self, latent_size: int, sentiment_size: int = 3) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(latent_size + sentiment_size, latent_size), nn.Tanh(), nn.Linear(latent_size, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(latent_size + sentiment_size),
            nn.Linear(latent_size + sentiment_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, latent: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, sentiment], dim=1)
        gate = torch.sigmoid(self.attention(x))
        x = torch.cat([latent * gate, sentiment], dim=1)
        return self.head(x).squeeze(-1)


class FinBERTVAEGANPredictor:
    def __init__(self, input_size: int, latent_size: int = 32, seed: int = 42, device: str | None = None) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.vae = FeatureVAE(input_size, latent_size).to(self.device)
        self.generator = LatentGenerator(noise_size=latent_size, latent_size=latent_size).to(self.device)
        self.discriminator = LatentDiscriminator(latent_size).to(self.device)
        self.predictor = PricePredictor(latent_size).to(self.device)
        self.vae_opt = torch.optim.AdamW(self.vae.parameters(), lr=1e-3, weight_decay=1e-4)
        self.g_opt = torch.optim.AdamW(self.generator.parameters(), lr=5e-4)
        self.d_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=5e-4)
        self.p_opt = torch.optim.AdamW(self.predictor.parameters(), lr=1e-3, weight_decay=1e-4)
        self.reg_loss = nn.SmoothL1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def fit(self, X: np.ndarray, sentiment: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32) -> TrainingHistory:
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        s_tensor = torch.as_tensor(sentiment, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        n_val = min(len(X_tensor) - 1, max(1, int(len(X_tensor) * 0.15)))
        n_train = len(X_tensor) - n_val
        loader = DataLoader(TensorDataset(X_tensor[:n_train], s_tensor[:n_train], y_tensor[:n_train]), batch_size=batch_size, shuffle=True)
        val_X = X_tensor[n_train:].to(self.device)
        val_s = s_tensor[n_train:].to(self.device)
        val_y = y_tensor[n_train:].to(self.device)
        history = TrainingHistory(vae_loss=[], predictor_loss=[], val_loss=[])
        best_state = None
        best_val = float("inf")
        stale = 0
        for _ in range(epochs):
            vae_losses, pred_losses = [], []
            for batch_X, batch_s, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_s = batch_s.to(self.device)
                batch_y = batch_y.to(self.device)

                recon, mu, logvar, real_latent = self.vae(batch_X)
                recon_loss = nn.functional.mse_loss(recon, batch_X)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = recon_loss + 0.001 * kl_loss
                self.vae_opt.zero_grad(set_to_none=True)
                vae_loss.backward(retain_graph=True)
                self.vae_opt.step()

                noise = torch.randn_like(real_latent)
                fake_latent = self.generator(noise).detach()
                d_loss = self.bce(self.discriminator(real_latent.detach()), torch.ones(len(batch_X), device=self.device))
                d_loss = d_loss + self.bce(self.discriminator(fake_latent), torch.zeros(len(batch_X), device=self.device))
                self.d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                self.d_opt.step()

                fake_latent = self.generator(torch.randn_like(real_latent))
                g_loss = self.bce(self.discriminator(fake_latent), torch.ones(len(batch_X), device=self.device))
                self.g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                self.g_opt.step()

                with torch.no_grad():
                    _, mu, _, _ = self.vae(batch_X)
                pred = self.predictor(mu, batch_s)
                pred_loss = self.reg_loss(pred, batch_y)
                self.p_opt.zero_grad(set_to_none=True)
                pred_loss.backward()
                self.p_opt.step()
                vae_losses.append(float(vae_loss.detach().cpu()))
                pred_losses.append(float(pred_loss.detach().cpu()))

            self.vae.eval()
            self.predictor.eval()
            with torch.no_grad():
                _, mu, _, _ = self.vae(val_X)
                val_loss = float(self.reg_loss(self.predictor(mu, val_s), val_y).detach().cpu())
            self.vae.train()
            self.predictor.train()
            history.vae_loss.append(float(np.mean(vae_losses)))
            history.predictor_loss.append(float(np.mean(pred_losses)))
            history.val_loss.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    "vae": {k: v.detach().cpu().clone() for k, v in self.vae.state_dict().items()},
                    "predictor": {k: v.detach().cpu().clone() for k, v in self.predictor.state_dict().items()},
                }
                stale = 0
            else:
                stale += 1
                if stale >= 12:
                    break
        if best_state is not None:
            self.vae.load_state_dict(best_state["vae"])
            self.predictor.load_state_dict(best_state["predictor"])
            self.vae.to(self.device)
            self.predictor.to(self.device)
        return history

    def predict(self, X: np.ndarray, sentiment: np.ndarray, batch_size: int = 256) -> np.ndarray:
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        s_tensor = torch.as_tensor(sentiment, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, s_tensor), batch_size=batch_size, shuffle=False)
        preds = []
        self.vae.eval()
        self.predictor.eval()
        with torch.no_grad():
            for batch_X, batch_s in loader:
                _, mu, _, _ = self.vae(batch_X.to(self.device))
                preds.append(self.predictor(mu, batch_s.to(self.device)).detach().cpu().numpy())
        return np.concatenate(preds)


def align_sentiment(sentiment: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    lookup_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    aligned = sentiment.reindex(lookup_dates).fillna({"sentiment_positive": 0.0, "sentiment_negative": 0.0, "sentiment_neutral": 1.0})
    return aligned[["sentiment_positive", "sentiment_negative", "sentiment_neutral"]].to_numpy(dtype=np.float32)


def run_finbert_vae_gan_benchmark(
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
    news_csv: str | Path | None = None,
    finbert_model: str = "ProsusAI/finbert",
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
    X_train, flat_names = flatten_lag_features(tensor_data.X_train, selection.features, lookback)
    X_test, _ = flatten_lag_features(tensor_data.X_test, selection.features, lookback)
    sentiment = load_finbert_sentiment(news_csv, features_df.index, model_name=finbert_model)
    s_train = align_sentiment(sentiment, tensor_data.train_base_dates)
    s_test = align_sentiment(sentiment, tensor_data.test_base_dates)
    model = FinBERTVAEGANPredictor(input_size=X_train.shape[1])
    history = model.fit(X_train, s_train, tensor_data.y_train, epochs=epochs)
    pred_scaled = model.predict(X_test, s_test)
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
            "model": "FinBERT-VAE-GAN",
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
    raw.to_csv(output_dir / f"{name}_raw.csv")
    selection.scores.to_csv(output_dir / f"{name}_feature_ic.csv", index=False)
    pd.DataFrame({"feature": flat_names}).to_csv(output_dir / f"{name}_flat_feature_map.csv", index=False)
    metrics = regression_metrics(actual, predicted, baseline)
    metrics["direction_accuracy_pct"] = direction_accuracy(
        pd.Series(actual, index=tensor_data.test_dates),
        pd.Series(predicted, index=tensor_data.test_dates),
        pd.Series(baseline, index=tensor_data.test_dates),
    )
    metrics["samples"] = int(len(actual))
    metrics["train_samples"] = int(len(tensor_data.y_train))
    metrics["model"] = "FinBERT-VAE-GAN"
    metrics["features"] = selection.features
    metrics["sentiment_source"] = str(news_csv) if news_csv else "neutral"
    metrics["final_vae_loss"] = history.vae_loss[-1]
    metrics["final_train_loss"] = history.predictor_loss[-1]
    metrics["final_val_loss"] = history.val_loss[-1]
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FinBERT-VAE-GAN benchmark forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", default="outputs_finbert_vae_gan_h21")
    parser.add_argument("--news-csv", default=None)
    parser.add_argument("--finbert-model", default="ProsusAI/finbert")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running FinBERT-VAE-GAN {name} ({ticker})...")
        result = run_finbert_vae_gan_benchmark(
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
            news_csv=args.news_csv,
            finbert_model=args.finbert_model,
        )
        results.append(result)
        print(f"{name}: RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}, MAPE={result['mape_pct']:.2f}%")
    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
