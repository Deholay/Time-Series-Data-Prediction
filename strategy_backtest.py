from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    std = returns.std(ddof=1)
    if len(returns) < 2 or std == 0 or pd.isna(std):
        return np.nan
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    rename_map = {c: str(c).strip().title() for c in raw.columns}
    raw = raw.rename(columns=rename_map)
    if "Adj Close" in raw.columns and "Close" not in raw.columns:
        raw["Close"] = raw["Adj Close"]
    return raw


def build_backtest_frame(
    index_name: str,
    predictions_path: Path,
    raw_path: Path,
    threshold_bps: float,
    cost_bps: float,
) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path, parse_dates=["date"])
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw = normalize_raw(raw)

    threshold = threshold_bps / 10_000.0
    cost = cost_bps / 10_000.0

    df = pred.merge(
        raw[["Open", "Close"]].rename(columns={"Open": "target_open", "Close": "target_raw_close"}),
        left_on="date",
        right_index=True,
        how="left",
    )
    if "forecast_base_date" not in df.columns:
        df["forecast_base_date"] = pd.NaT

    df["predicted_return"] = df["predicted_close"] / df["naive_previous_close"] - 1.0
    df["actual_close_to_close_return"] = df["actual_close"] / df["naive_previous_close"] - 1.0
    df["actual_open_to_close_return"] = df["target_raw_close"] / df["target_open"] - 1.0

    df["signal"] = np.select(
        [df["predicted_return"] > threshold, df["predicted_return"] < -threshold],
        [1, -1],
        default=0,
    )
    df["trade_cost"] = np.where(df["signal"] != 0, cost, 0.0)
    df["strategy_close_to_close_return"] = df["signal"] * df["actual_close_to_close_return"] - df["trade_cost"]
    df["strategy_open_to_close_return"] = df["signal"] * df["actual_open_to_close_return"] - df["trade_cost"]
    df["benchmark_close_to_close_return"] = df["actual_close_to_close_return"]
    df["benchmark_open_to_close_return"] = df["actual_open_to_close_return"]
    df["strategy_close_to_close_equity"] = (1.0 + df["strategy_close_to_close_return"].fillna(0.0)).cumprod()
    df["strategy_open_to_close_equity"] = (1.0 + df["strategy_open_to_close_return"].fillna(0.0)).cumprod()
    df["benchmark_close_to_close_equity"] = (1.0 + df["benchmark_close_to_close_return"].fillna(0.0)).cumprod()
    df["benchmark_open_to_close_equity"] = (1.0 + df["benchmark_open_to_close_return"].fillna(0.0)).cumprod()
    df["index"] = index_name
    df["threshold_bps"] = threshold_bps
    df["cost_bps"] = cost_bps
    return df


def summarize_backtest(df: pd.DataFrame) -> dict[str, float | int | str]:
    traded = df[df["signal"] != 0]
    actual_direction = np.sign(df["actual_close_to_close_return"])
    predicted_direction = np.sign(df["signal"])
    active_mask = df["signal"] != 0

    ic = df["predicted_return"].corr(df["actual_close_to_close_return"], method="pearson")
    rank_ic = df["predicted_return"].corr(df["actual_close_to_close_return"], method="spearman")
    active_hit_rate = (
        float((predicted_direction[active_mask] == actual_direction[active_mask]).mean() * 100.0)
        if active_mask.any()
        else np.nan
    )

    return {
        "index": str(df["index"].iloc[0]),
        "threshold_bps": float(df["threshold_bps"].iloc[0]),
        "cost_bps": float(df["cost_bps"].iloc[0]),
        "samples": int(len(df)),
        "trades": int((df["signal"] != 0).sum()),
        "long_days": int((df["signal"] > 0).sum()),
        "short_days": int((df["signal"] < 0).sum()),
        "cash_days": int((df["signal"] == 0).sum()),
        "predicted_return_mean_pct": float(df["predicted_return"].mean() * 100.0),
        "actual_return_mean_pct": float(df["actual_close_to_close_return"].mean() * 100.0),
        "pearson_ic": float(ic) if pd.notna(ic) else np.nan,
        "spearman_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
        "active_hit_rate_pct": active_hit_rate,
        "strategy_cc_total_return_pct": float((df["strategy_close_to_close_equity"].iloc[-1] - 1.0) * 100.0),
        "benchmark_cc_total_return_pct": float((df["benchmark_close_to_close_equity"].iloc[-1] - 1.0) * 100.0),
        "strategy_oc_total_return_pct": float((df["strategy_open_to_close_equity"].iloc[-1] - 1.0) * 100.0),
        "benchmark_oc_total_return_pct": float((df["benchmark_open_to_close_equity"].iloc[-1] - 1.0) * 100.0),
        "strategy_oc_sharpe": annualized_sharpe(df["strategy_open_to_close_return"]),
        "benchmark_oc_sharpe": annualized_sharpe(df["benchmark_open_to_close_return"]),
        "strategy_oc_max_drawdown_pct": max_drawdown(df["strategy_open_to_close_return"]) * 100.0,
        "benchmark_oc_max_drawdown_pct": max_drawdown(df["benchmark_open_to_close_return"]) * 100.0,
        "strategy_oc_win_rate_pct": float((traded["strategy_open_to_close_return"] > 0).mean() * 100.0)
        if len(traded)
        else np.nan,
        "avg_trade_oc_return_pct": float(traded["strategy_open_to_close_return"].mean() * 100.0)
        if len(traded)
        else np.nan,
    }


def plot_equity_curves(index_name: str, df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    ax.set_facecolor("white")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(
        df["date"],
        (df["strategy_open_to_close_equity"] - 1.0) * 100.0,
        color="#d28b26",
        linewidth=2.0,
        label="Strategy open-to-close",
    )
    ax.plot(
        df["date"],
        (df["benchmark_open_to_close_equity"] - 1.0) * 100.0,
        color="#30343b",
        linewidth=2.0,
        label="Always-long open-to-close",
    )
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.set_title(
        f"{index_name} Single-Day Strategy Backtest",
        fontsize=14,
        color="#20242a",
        loc="left",
    )
    ax.set_ylabel("Compounded return (%)")
    ax.set_xlabel("Trade date")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest one-day trading signals from LSTM prediction CSVs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--backtest-dir", default="backtests_h1")
    parser.add_argument("--threshold-bps", type=float, nargs="+", default=[0, 5, 10, 15, 20, 50])
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--plot-threshold-bps", type=float, default=10.0)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    backtest_dir = Path(args.backtest_dir)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(outputs_dir / "benchmark_summary.csv")
    all_summaries = []

    for index_name in summary["index"]:
        for threshold_bps in args.threshold_bps:
            df = build_backtest_frame(
                index_name=index_name,
                predictions_path=outputs_dir / f"{index_name}_predictions.csv",
                raw_path=outputs_dir / f"{index_name}_raw.csv",
                threshold_bps=threshold_bps,
                cost_bps=args.cost_bps,
            )
            detail_path = backtest_dir / f"{index_name}_strategy_threshold_{threshold_bps:g}bps.csv"
            df.to_csv(detail_path, index=False)
            all_summaries.append(summarize_backtest(df))

            if threshold_bps == args.plot_threshold_bps:
                plot_equity_curves(
                    index_name=index_name,
                    df=df,
                    output_path=backtest_dir / f"{index_name}_strategy_threshold_{threshold_bps:g}bps.png",
                )

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(backtest_dir / "strategy_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved backtest outputs to {backtest_dir.resolve()}")


if __name__ == "__main__":
    main()
