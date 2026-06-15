from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from indicator import add_technical_indicators
from tensor_transform import build_lstm_tensors


PALETTE = {
    "actual": "#1f5a99",
    "predicted": "#d28b26",
    "strategy": "#8a4f9e",
    "benchmark": "#30343b",
    "grid": "#d9dee7",
}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#30343b", labelsize=9)


def plot_tensor_surface(
    index_name: str,
    raw_path: Path,
    feature_ic_path: Path,
    output_path: Path,
    lookback: int,
    horizon: int,
    top_k: int,
    train_end: str,
    test_start: str,
    test_end: str,
) -> None:
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    features_df = add_technical_indicators(raw)
    selected_features = pd.read_csv(feature_ic_path)["feature"].head(top_k).tolist()
    tensor_data = build_lstm_tensors(
        features_df,
        feature_columns=selected_features,
        lookback=lookback,
        horizon=horizon,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        target_mode="log_return",
    )
    feature_map = pd.DataFrame(
        {"feature_channel": np.arange(1, len(selected_features) + 1), "feature": selected_features}
    )
    feature_map.to_csv(output_path.with_name(output_path.stem + "_feature_map.csv"), index=False)

    sample = tensor_data.X_test[0]
    time_axis = np.arange(1, sample.shape[0] + 1)
    feature_axis = np.arange(1, sample.shape[1] + 1)
    x_grid, y_grid = np.meshgrid(feature_axis, time_axis)

    fig = plt.figure(figsize=(11, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        sample,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )
    ax.set_title(
        f"{index_name} LSTM Input Tensor Sample",
        fontsize=14,
        color="#20242a",
        pad=16,
    )
    ax.text2D(
        0.02,
        0.94,
        f"Tensor shape: {tensor_data.X_test.shape} = samples x {lookback} time steps x {len(selected_features)} features",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    ax.set_xlabel("Feature channel", labelpad=10)
    ax.set_ylabel("Lookback time step", labelpad=10)
    ax.set_zlabel("Scaled feature value", labelpad=10)
    ax.set_xticks(feature_axis)
    ax.set_xticklabels([str(i) for i in feature_axis], fontsize=8)
    ax.view_init(elev=28, azim=-132)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label="Standardized value")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.07)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_price_predictions(index_name: str, prediction_path: Path, output_path: Path) -> None:
    df = pd.read_csv(prediction_path, parse_dates=["date"]).sort_values("date")
    horizon = int(df["horizon_trading_days"].iloc[0]) if "horizon_trading_days" in df.columns else 1
    horizon_label = "Next-Day" if horizon == 1 else f"{horizon}-Trading-Day Ahead"
    model_label = str(df["model"].iloc[0]) if "model" in df.columns else "LSTM"
    start_date = df["date"].dt.normalize().min()
    end_date = df["date"].dt.normalize().max()
    x_slots = (df["date"].dt.normalize() - start_date).dt.days.to_numpy()
    all_days = pd.date_range(start_date, end_date, freq="D")
    month_starts = pd.date_range(start_date.normalize().replace(day=1), end_date, freq="MS")
    month_starts = month_starts[month_starts >= start_date]
    month_slots = (month_starts - start_date).days

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    _style_axes(ax)
    ax.plot(
        x_slots,
        df["actual_close"],
        color=PALETTE["actual"],
        linewidth=2.2,
        label="Actual index close",
    )
    ax.plot(
        x_slots,
        df["predicted_close"],
        color=PALETTE["predicted"],
        linewidth=2.0,
        linestyle="--",
        label=f"Daily {model_label} predicted close",
    )
    ax.plot(
        x_slots,
        df["naive_previous_close"],
        color=PALETTE["benchmark"],
        linewidth=1.4,
        alpha=0.55,
        label="Previous-close baseline",
    )

    ax.set_title(
        f"{index_name} {horizon_label} Predicted Close vs Actual Index Close",
        fontsize=14,
        color="#20242a",
        loc="left",
    )
    ax.set_ylabel("Index close")
    ax.set_xlabel("Prediction date (one calendar day per slot)")
    ax.set_xlim(-0.5, len(all_days) - 0.5)
    ax.set_xticks(month_slots)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in month_starts])
    ax.set_xticks(np.arange(len(all_days)), minor=True)
    ax.grid(True, axis="y", which="major", color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="x", which="minor", color=PALETTE["grid"], linewidth=0.35, alpha=0.45)
    ax.grid(True, axis="x", which="major", color=PALETTE["grid"], linewidth=0.9, alpha=0.95)
    ax.legend(frameon=False, loc="best")

    rmse = np.sqrt(np.mean((df["predicted_close"] - df["actual_close"]) ** 2))
    mae = np.mean(np.abs(df["predicted_close"] - df["actual_close"]))
    final_text = (
        f"Last day: actual {df['actual_close'].iloc[-1]:,.2f} | "
        f"predicted {df['predicted_close'].iloc[-1]:,.2f} | "
        f"RMSE {rmse:,.2f} | MAE {mae:,.2f}"
    )
    ax.text(
        0.01,
        -0.18,
        final_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accumulated_returns(index_name: str, prediction_path: Path, output_path: Path) -> None:
    """Backward-compatible wrapper. The requested report plot is now price vs price."""
    plot_price_predictions(index_name, prediction_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LSTM tensor and daily price prediction charts with Matplotlib.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(outputs_dir / "benchmark_summary.csv")
    for index_name in summary["index"]:
        plot_tensor_surface(
            index_name=index_name,
            raw_path=outputs_dir / f"{index_name}_raw.csv",
            feature_ic_path=outputs_dir / f"{index_name}_feature_ic.csv",
            output_path=plots_dir / f"{index_name}_tensor_3d.png",
            lookback=args.lookback,
            horizon=args.horizon,
            top_k=args.top_k,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        plot_price_predictions(
            index_name=index_name,
            prediction_path=outputs_dir / f"{index_name}_predictions.csv",
            output_path=plots_dir / f"{index_name}_price_prediction.png",
        )

    print(f"Saved plots to {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
