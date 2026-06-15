from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class FeatureSelectionResult:
    features: list[str]
    scores: pd.DataFrame


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standard OHLCV column names from common Yahoo formats."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
    }
    out = out.rename(columns={c: rename_map.get(str(c).strip().lower(), c) for c in out.columns})

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    missing = [c for c in ("Open", "High", "Low", "Close") if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out.sort_index()
    return out


def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    ema_short = exponential_moving_average(close, short_window)
    ema_long = exponential_moving_average(close, long_window)
    macd_line = ema_short - ema_long
    signal = macd_line.ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_signal": signal,
            "MACD_hist": macd_line - signal,
        },
        index=close.index,
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs))
    return value.fillna(100.0).where(avg_gain.notna(), np.nan)


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    return ((close - lowest_low) / denominator) * 100.0


def add_technical_indicators(
    df: pd.DataFrame,
    ma_windows: Iterable[int] = (5, 10, 20, 60),
    ema_windows: Iterable[int] = (12, 26),
    rsi_windows: Iterable[int] = (6, 12, 14),
    stochastic_windows: Iterable[int] = (14,),
) -> pd.DataFrame:
    """Add the indicators described in the template plus common return/volatility channels."""
    out = normalize_ohlcv_columns(df)
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)

    out["Return_1d"] = close.pct_change()
    out["LogReturn_1d"] = np.log(close / close.shift(1))
    out["HL_range_pct"] = (high - low) / close.replace(0.0, np.nan)
    out["OC_change_pct"] = (close - out["Open"].astype(float)) / out["Open"].replace(0.0, np.nan)
    out["Volatility_10"] = out["LogReturn_1d"].rolling(10, min_periods=10).std()

    for window in ma_windows:
        out[f"MA_{window}"] = moving_average(close, window)
        out[f"Close_to_MA_{window}"] = close / out[f"MA_{window}"] - 1.0

    for window in ema_windows:
        out[f"EMA_{window}"] = exponential_moving_average(close, window)
        out[f"Close_to_EMA_{window}"] = close / out[f"EMA_{window}"] - 1.0

    out = pd.concat([out, macd(close)], axis=1)

    for window in rsi_windows:
        out[f"RSI_{window}"] = rsi(close, window)

    for window in stochastic_windows:
        out[f"STOCH_K_{window}"] = stochastic_k(high, low, close, window)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def select_features_by_ic(
    df: pd.DataFrame,
    horizon: int = 1,
    top_k: int = 16,
    candidate_columns: Iterable[str] | None = None,
    target_col: str = "Close",
) -> FeatureSelectionResult:
    """Rank features by Pearson IC against the future close at the requested horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    numeric = df.select_dtypes(include=[np.number])
    if candidate_columns is None:
        excluded = {target_col, "Adj Close"}
        candidates = [c for c in numeric.columns if c not in excluded]
    else:
        candidates = [c for c in candidate_columns if c in numeric.columns]

    target = numeric[target_col].shift(-horizon)
    rows = []
    for col in candidates:
        pair = pd.concat([numeric[col], target], axis=1).dropna()
        if len(pair) < 30 or pair.iloc[:, 0].nunique() <= 1:
            corr = np.nan
        else:
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        rows.append({"feature": col, "ic": corr, "abs_ic": abs(corr) if pd.notna(corr) else np.nan})

    scores = pd.DataFrame(rows).dropna().sort_values("abs_ic", ascending=False).reset_index(drop=True)
    features = scores.head(top_k)["feature"].tolist()
    return FeatureSelectionResult(features=features, scores=scores)
