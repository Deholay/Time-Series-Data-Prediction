from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from benchmark_utils import INDEX_TICKERS, direction_accuracy, regression_metrics
from indicator import normalize_ohlcv_columns


@dataclass(frozen=True)
class LLMTIMEConfig:
    precision: int = 2
    alpha: float = 0.95
    style: str = "gpt"
    separator: str = ","


@dataclass
class LLMTIMEScaler:
    offset: float
    scale: float
    precision: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.offset) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.offset


@dataclass(frozen=True)
class HuggingFaceConfig:
    model: str = "distilgpt2"
    device: str = "auto"
    max_new_tokens: int = 32


def fit_llmtime_scaler(values: np.ndarray, precision: int = 2, alpha: float = 0.95) -> LLMTIMEScaler:
    values = np.asarray(values, dtype=float)
    offset = float(np.median(values))
    centered = np.abs(values - offset)
    scale = float(np.quantile(centered, alpha))
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return LLMTIMEScaler(offset=offset, scale=scale, precision=precision)


def _format_scaled_number(value: float, precision: int) -> str:
    scaled_int = int(np.trunc(value * (10**precision)))
    sign = "-" if scaled_int < 0 else ""
    digits = str(abs(scaled_int)).zfill(precision + 1)
    return sign + digits


def encode_llmtime_values(values: np.ndarray, scaler: LLMTIMEScaler, style: str = "gpt", separator: str = ",") -> str:
    encoded = []
    for value in scaler.transform(values):
        token = _format_scaled_number(float(value), scaler.precision)
        if style.lower() == "gpt":
            token = " ".join(token)
        encoded.append(token)
    return separator.join(encoded) + separator


def decode_llmtime_text(text: str, scaler: LLMTIMEScaler, style: str = "gpt") -> list[float]:
    if style.lower() == "gpt":
        compact = text.replace(" ", "")
    else:
        compact = text
    raw_tokens = [t for t in re.split(r"[,;\s]+", compact) if t]
    values = []
    for token in raw_tokens:
        if not re.fullmatch(r"-?\d+", token):
            continue
        sign = -1.0 if token.startswith("-") else 1.0
        digits = token[1:] if token.startswith("-") else token
        scaled = sign * (int(digits) / (10**scaler.precision))
        values.append(float(scaler.inverse_transform(np.array([scaled]))[0]))
    return values


def build_llmtime_prompt(
    history_values: np.ndarray,
    scaler: LLMTIMEScaler,
    horizon: int,
    config: LLMTIMEConfig,
) -> str:
    encoded = encode_llmtime_values(history_values, scaler=scaler, style=config.style, separator=config.separator)
    return (
        "Continue the numeric time series. Return only the next encoded value, "
        f"using the same digit format and separator. Forecast horizon: {horizon} trading days.\n"
        f"Series: {encoded}"
    )


def local_trend_backend(history_values: np.ndarray, horizon: int) -> float:
    """Deterministic local fallback: extrapolate a linear trend over the lookback window."""
    values = np.asarray(history_values, dtype=float)
    if len(values) < 2:
        return float(values[-1])
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, deg=1)
    return float(intercept + slope * (len(values) - 1 + horizon))


def call_openai_compatible_backend(prompt: str) -> str:
    api_url = os.environ["LLMTIME_API_URL"]
    api_key = os.environ.get("LLMTIME_API_KEY", "")
    model = os.environ.get("LLMTIME_MODEL", "")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


@lru_cache(maxsize=2)
def _load_huggingface_model(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "LLMTIME Hugging Face backend requires transformers and PyTorch. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def call_huggingface_backend(prompt: str, hf_config: HuggingFaceConfig | None = None) -> str:
    hf_config = hf_config or HuggingFaceConfig(
        model=os.environ.get("LLMTIME_HF_MODEL", "distilgpt2"),
        device=os.environ.get("LLMTIME_HF_DEVICE", "auto"),
        max_new_tokens=int(os.environ.get("LLMTIME_HF_MAX_NEW_TOKENS", "32")),
    )
    tokenizer, model, device = _load_huggingface_model(hf_config.model, hf_config.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for Hugging Face generation.") from exc

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=hf_config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def predict_llmtime_value(
    history_values: np.ndarray,
    scaler: LLMTIMEScaler,
    horizon: int,
    config: LLMTIMEConfig,
    backend: str = "local",
    hf_config: HuggingFaceConfig | None = None,
) -> tuple[float, str]:
    if backend == "local":
        predicted = local_trend_backend(history_values, horizon=horizon)
        prompt = build_llmtime_prompt(history_values, scaler=scaler, horizon=horizon, config=config)
        return predicted, prompt
    if backend == "api":
        prompt = build_llmtime_prompt(history_values, scaler=scaler, horizon=horizon, config=config)
        completion = call_openai_compatible_backend(prompt)
        decoded = decode_llmtime_text(completion, scaler=scaler, style=config.style)
        if not decoded:
            raise ValueError(f"Could not decode LLMTIME API completion: {completion!r}")
        return decoded[0], prompt
    if backend == "hf":
        prompt = build_llmtime_prompt(history_values, scaler=scaler, horizon=horizon, config=config)
        completion = call_huggingface_backend(prompt, hf_config=hf_config)
        decoded = decode_llmtime_text(completion, scaler=scaler, style=config.style)
        if not decoded:
            raise ValueError(f"Could not decode LLMTIME Hugging Face completion: {completion!r}")
        return decoded[0], prompt
    raise ValueError("backend must be 'local', 'api', or 'hf'")


def download_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    return normalize_ohlcv_columns(df)


def run_llmtime_benchmark(
    name: str,
    ticker: str,
    start: str,
    end: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lookback: int,
    horizon: int,
    output_dir: Path,
    backend: str = "local",
    config: LLMTIMEConfig | None = None,
    hf_config: HuggingFaceConfig | None = None,
) -> dict:
    config = config or LLMTIMEConfig()
    raw = download_index(ticker, start=start, end=end)
    data = raw.sort_index().replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))

    train_values = data.loc[:train_end, "Close"].to_numpy(dtype=float)
    scaler = fit_llmtime_scaler(train_values, precision=config.precision, alpha=config.alpha)

    rows = []
    first_prompt = ""
    close = data["Close"].to_numpy(dtype=float)
    dates = data.index
    for end_idx in range(lookback - 1, len(data) - horizon):
        target_idx = end_idx + horizon
        pred_date = dates[target_idx]
        if pred_date < pd.Timestamp(test_start) or pred_date > pd.Timestamp(test_end):
            continue
        history = close[end_idx - lookback + 1 : end_idx + 1]
        predicted_close, prompt = predict_llmtime_value(
            history,
            scaler=scaler,
            horizon=horizon,
            config=config,
            backend=backend,
            hf_config=hf_config,
        )
        if not first_prompt:
            first_prompt = prompt
        base = close[end_idx]
        actual = close[target_idx]
        rows.append(
            {
                "forecast_base_date": dates[end_idx],
                "date": pred_date,
                "index": name,
                "ticker": ticker,
                "model": f"LLMTIME-{backend}",
                "horizon_trading_days": horizon,
                "actual_close": actual,
                "predicted_close": predicted_close,
                "naive_previous_close": base,
                "actual_log_return": np.log(actual / base),
                "predicted_log_return": np.log(predicted_close / base) if predicted_close > 0 else np.nan,
                "absolute_error": abs(predicted_close - actual),
                "pct_error": abs((predicted_close - actual) / actual) * 100.0,
            }
        )

    prediction_df = pd.DataFrame(rows)
    if prediction_df.empty:
        raise ValueError("No LLMTIME test predictions were generated")
    prediction_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    raw.to_csv(output_dir / f"{name}_raw.csv")
    (output_dir / f"{name}_first_prompt.txt").write_text(first_prompt, encoding="utf-8")

    actual = prediction_df["actual_close"].to_numpy()
    predicted = prediction_df["predicted_close"].to_numpy()
    baseline = prediction_df["naive_previous_close"].to_numpy()
    metrics = regression_metrics(actual, predicted, baseline)
    metrics["direction_accuracy_pct"] = direction_accuracy(
        prediction_df["actual_close"],
        prediction_df["predicted_close"],
        prediction_df["naive_previous_close"],
    )
    metrics["samples"] = int(len(prediction_df))
    metrics["train_samples"] = int(len(train_values))
    metrics["model"] = f"LLMTIME-{backend}"
    metrics["features"] = ["Close"]
    metrics["tokenizer"] = config.style
    metrics["precision"] = config.precision
    metrics["alpha"] = config.alpha
    if backend == "hf":
        resolved_hf = hf_config or HuggingFaceConfig(
            model=os.environ.get("LLMTIME_HF_MODEL", "distilgpt2"),
            device=os.environ.get("LLMTIME_HF_DEVICE", "auto"),
            max_new_tokens=int(os.environ.get("LLMTIME_HF_MAX_NEW_TOKENS", "32")),
        )
        metrics["hf_model"] = resolved_hf.model
    return {"index": name, "ticker": ticker, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLMTIME tokenization benchmark forecasts.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs_llmtime_h1")
    parser.add_argument("--backend", choices=["local", "api", "hf"], default="hf")
    parser.add_argument("--style", choices=["gpt", "llama"], default="gpt")
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--hf-model", default=os.environ.get("LLMTIME_HF_MODEL", "distilgpt2"))
    parser.add_argument("--hf-device", default=os.environ.get("LLMTIME_HF_DEVICE", "auto"))
    parser.add_argument("--hf-max-new-tokens", type=int, default=int(os.environ.get("LLMTIME_HF_MAX_NEW_TOKENS", "32")))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LLMTIMEConfig(precision=args.precision, alpha=args.alpha, style=args.style)
    hf_config = HuggingFaceConfig(
        model=args.hf_model,
        device=args.hf_device,
        max_new_tokens=args.hf_max_new_tokens,
    )

    results = []
    for name, ticker in INDEX_TICKERS.items():
        print(f"Running LLMTIME-{args.backend} {name} ({ticker})...")
        result = run_llmtime_benchmark(
            name=name,
            ticker=ticker,
            start=args.start,
            end=args.end,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            lookback=args.lookback,
            horizon=args.horizon,
            output_dir=output_dir,
            backend=args.backend,
            config=config,
            hf_config=hf_config if args.backend == "hf" else None,
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
