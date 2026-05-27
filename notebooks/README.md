# notebooks/

Exploratory analysis and time-series diagnostics for the EV Charging Smart System.

## Files

### `eda.py`
Runs as a standalone script; generates 4 PNG charts saved to `models/`:

| Chart | Description |
|---|---|
| `eda_distributions.png` | Wait time, session duration, energy-per-session histograms |
| `eda_temporal.png` | Hourly, day-of-week, monthly and weekday-vs-weekend wait patterns |
| `eda_stations.png` | Station load vs average wait scatter + station-level distribution |
| `eda_correlation.png` | Feature correlation matrix (top 12 numeric features) |

Run:
```bash
PYTHONPATH=.. python eda.py
```

---

### `time_series_analysis.py`
Time-series diagnostics for the LSTM training data:

| Output | Description |
|---|---|
| `ts_raw_series.png`    | Raw hourly wait-time series for the busiest station |
| `ts_decomposition.png` | Seasonal decomposition: trend + 24h seasonal + residual |
| `ts_acf_pacf.png`      | ACF and PACF plots (48-lag) |
| `ts_rolling_lag.png`   | 24-hour rolling mean/std and lag-1 scatter plot |

Also prints the **ADF stationarity test** result and **LSTM sequence statistics**.

Run:
```bash
PYTHONPATH=.. python time_series_analysis.py
```

---

## Converting to Jupyter Notebooks

```bash
pip install jupytext
jupytext --to notebook eda.py
jupytext --to notebook time_series_analysis.py
jupyter notebook eda.ipynb
```

## Prerequisites

```bash
pip install matplotlib seaborn statsmodels pandas numpy
```
Run `python src/train.py` first to ensure `data/` and `models/` are populated.
