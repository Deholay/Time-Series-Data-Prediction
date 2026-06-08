# TS Prediction Pipeline

This project builds a technical-indicator based LSTM forecasting pipeline for Nasdaq, S&P 500, and SOX index data.

The workflow follows the PDF template:

1. Download OHLCV market data.
2. Generate technical indicators.
3. Select indicators by Pearson information coefficient.
4. Transform features into a 3-D LSTM tensor.
5. Train an LSTM through December 2025.
6. Predict January 2026 through May 2026.
7. Benchmark against a naive previous-close baseline.
8. Plot the input tensor and daily predicted close against the actual index close.

## Files

| File | Purpose |
|---|---|
| `indicator.py` | Calculates MA, EMA, MACD, RSI, Stochastic %K, return, range, and volatility features. Also ranks features by Pearson IC. |
| `tensor_transform.py` | Converts selected features into LSTM tensors shaped as `(batch, time_steps, input_features)`. |
| `LSTM.py` | Defines the PyTorch LSTM model, training loop, early stopping, and prediction helper. |
| `benchmark.py` | Downloads `^IXIC`, `^GSPC`, and `^SOX`, trains models, saves predictions and benchmark metrics. |
| `plot_results.py` | Uses Matplotlib to plot 3-D tensor samples and daily predicted close vs actual index close. |
| `strategy_backtest.py` | Converts one-day predictions into long/short/cash trading signals and tests close-to-close plus open-to-close PnL. |
| `TS_Prediction_Pipeline.ipynb` | Notebook version of the full pipeline. It imports the scripts above instead of reimplementing them. |
| `requirements.txt` | Python package dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

The code was verified with Python 3.13, PyTorch 2.12, pandas 3.0, and scikit-learn 1.9.

## Run The Full Benchmark

```bash
python benchmark.py --output-dir outputs
```

Default settings:

| Setting | Value |
|---|---|
| Data start | `2010-01-01` |
| Data end | `2026-06-01` |
| Training end | `2025-12-31` |
| Prediction period | `2026-01-01` to `2026-05-31` |
| Lookback window | `30` trading days |
| Prediction horizon | `1` trading day |
| Selected features | top `16` by Pearson IC |
| Training target | next-day log return |

The model predicts next-day log return, then converts the prediction back to close price:

```text
predicted_close = previous_close * exp(predicted_log_return)
```

This is more stable than directly forecasting the raw close-price level.

## Run One-Month-Ahead Daily Rolling Forecast

Use `--horizon 21` to predict about one trading month ahead while still updating the input window every trading day:

```bash
python benchmark.py --horizon 21 --output-dir outputs_h21
python plot_results.py --outputs-dir outputs_h21 --plots-dir plots_h21 --horizon 21
```

For example, the first 2026 prediction row uses `forecast_base_date = 2025-12-02` to predict the target close on `2026-01-02`. The next row uses `2025-12-03` to predict `2026-01-05`, so the forecast is still daily rolling.

## Generate Plots

```bash
python plot_results.py --outputs-dir outputs --plots-dir plots
```

## Run Single-Day Strategy Backtest

The strategy test uses `horizon=1` prediction CSVs. Signals are generated after the base-day close, then the more tradable PnL path enters at the next day's open and exits at that day's close.

```bash
python benchmark.py --horizon 1 --output-dir outputs
python strategy_backtest.py --outputs-dir outputs --backtest-dir backtests_h1 --threshold-bps 0 5 10 15 20 50 --cost-bps 2 --plot-threshold-bps 10
```

Signal rule:

```text
long  if predicted_return > +threshold
short if predicted_return < -threshold
cash  otherwise
```

The script also writes a `strategy_summary.csv` and per-index detail CSVs for each threshold.

Generated plot types:

| Plot | Description |
|---|---|
| `*_tensor_3d.png` | 3-D surface view of one LSTM input tensor sample. |
| `*_tensor_3d_feature_map.csv` | Mapping from feature channel number to feature name. |
| `*_price_prediction.png` | Daily predicted close vs actual index close, with the previous-close baseline. |

## Notebook

Open:

```text
TS_Prediction_Pipeline.ipynb
```

The notebook runs the same pipeline by importing the project scripts. Set:

```python
RUN_TRAINING = True
```

to retrain the models. Set it to `False` to reuse existing CSV outputs and regenerate plots only.

## Outputs

`outputs/` contains:

| File Pattern | Description |
|---|---|
| `benchmark_summary.csv` | Metrics for all indices. |
| `benchmark_summary.json` | Same metrics in JSON form. |
| `*_raw.csv` | Downloaded market data. |
| `*_feature_ic.csv` | Pearson IC feature ranking. |
| `*_predictions.csv` | Daily actual close, predicted close, previous-close baseline, returns, and errors. |

`plots/` contains:

| File Pattern | Description |
|---|---|
| `*_tensor_3d.png` | 3-D tensor visualization. |
| `*_price_prediction.png` | Daily predicted close vs actual index close chart. |

## Benchmark Metrics

The benchmark compares predicted close against the actual close for January through May 2026.

Metrics:

| Metric | Meaning |
|---|---|
| `rmse` | Root mean squared error of LSTM close prediction. |
| `mae` | Mean absolute error of LSTM close prediction. |
| `mape_pct` | Mean absolute percentage error. |
| `naive_rmse` | RMSE for previous-close baseline. |
| `naive_mae` | MAE for previous-close baseline. |
| `rmse_vs_naive_pct` | Positive value means LSTM improves over baseline RMSE. |
| `mae_vs_naive_pct` | Positive value means LSTM improves over baseline MAE. |
| `direction_accuracy_pct` | Percentage of days where predicted direction matches actual direction. |

## Pipeline Details

### Indicator Layer

`indicator.py` adds:

- `MA_5`, `MA_10`, `MA_20`, `MA_60`
- `EMA_12`, `EMA_26`
- `MACD`, `MACD_signal`, `MACD_hist`
- `RSI_6`, `RSI_12`, `RSI_14`
- `STOCH_K_14`
- daily return, log return, high-low range, open-close change, 10-day volatility

Feature selection is performed only on the training period to avoid leakage.

### Tensor Layer

`tensor_transform.py` creates:

```text
X_train: (training_samples, lookback, selected_features)
X_test:  (test_samples, lookback, selected_features)
y_train: scaled next-day log return
y_test:  scaled next-day log return
```

Feature and target scalers are fit on training data only.

### Model Layer

`LSTM.py` uses:

- PyTorch `nn.LSTM`
- layer normalization
- dense prediction head
- AdamW optimizer
- Smooth L1 loss
- gradient clipping
- early stopping on validation loss

## Reproducible Command Sequence

```bash
pip install -r requirements.txt
python benchmark.py --output-dir outputs
python plot_results.py --outputs-dir outputs --plots-dir plots
```
