# notebooks/time_series_analysis.py
# ─── Time-Series Decomposition & Stationarity Analysis ───────────────────────

# %% [markdown]
# # 📈 EV Charging — Time Series Analysis
# Stationarity tests, seasonal decomposition, autocorrelation, and
# sequence data preparation for the LSTM model.

# %% Setup
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config.config import RAW_DIR, PROC_DIR, MODEL_DIR

STYLE = dict(bg="#0d1117", panel="#161b22", green="#39d353",
             blue="#58a6ff", orange="#f78166", text="#c9d1d9")
plt.rcParams.update({
    "axes.facecolor": STYLE["panel"], "figure.facecolor": STYLE["bg"],
    "text.color": STYLE["text"],      "axes.labelcolor": STYLE["text"],
    "xtick.color": STYLE["text"],     "ytick.color": STYLE["text"],
    "axes.edgecolor": "#21262d",      "grid.color": "#21262d",
    "axes.grid": True,
})

# %% Load and build hourly series
sessions = pd.read_csv(RAW_DIR / "charging_sessions.csv", parse_dates=["start_time"])

# Aggregate to hourly mean wait per station (pick top station by volume)
top_station = (sessions.groupby("station_id")["wait_time_minutes"]
               .count().idxmax())
station_df  = sessions[sessions["station_id"] == top_station].copy()
station_df  = station_df.set_index("start_time").sort_index()
hourly      = station_df["wait_time_minutes"].resample("H").mean().fillna(method="ffill")

print(f"Station {top_station}: {len(hourly)} hourly observations")
print(hourly.describe())

# %% Plot raw series
fig, ax = plt.subplots(figsize=(14, 4), facecolor=STYLE["bg"])
ax.plot(hourly.index, hourly.values, color=STYLE["blue"], lw=0.8, alpha=0.9)
ax.fill_between(hourly.index, 0, hourly.values, color=STYLE["blue"], alpha=0.1)
ax.set_xlabel("Date")
ax.set_ylabel("Avg Wait (min)")
ax.set_title(f"Hourly Wait Time — Station {top_station}")
plt.tight_layout()
fig.savefig(MODEL_DIR / "ts_raw_series.png", dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Saved: ts_raw_series.png")

# %% ADF stationarity test
try:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(hourly.dropna(), autolag="AIC")
    print(f"\n── ADF Stationarity Test ──────────────────")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Stationary    : {'Yes ✅' if result[1] < 0.05 else 'No ❌'}")
    for key, val in result[4].items():
        print(f"  Critical ({key}): {val:.4f}")
except ImportError:
    print("statsmodels not installed — skipping ADF test")

# %% Seasonal decomposition
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    # Require at least 2 periods of data (48 hours)
    if len(hourly.dropna()) >= 48:
        dec = seasonal_decompose(hourly.dropna(), model="additive", period=24)

        fig = plt.figure(figsize=(14, 10), facecolor=STYLE["bg"])
        gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5)

        components = [
            ("Original",     hourly.dropna(),   STYLE["blue"]),
            ("Trend",        dec.trend,         STYLE["green"]),
            ("Seasonal",     dec.seasonal,      STYLE["orange"]),
            ("Residual",     dec.resid,         STYLE["text"]),
        ]
        for i, (name, data, color) in enumerate(components):
            ax = fig.add_subplot(gs[i])
            ax.plot(data.index, data.values, color=color, lw=1.2)
            ax.set_ylabel(name, fontsize=9)
            ax.tick_params(labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")

        fig.suptitle("Seasonal Decomposition (Period = 24h)",
                     color=STYLE["text"], fontsize=12, y=1.01)
        fig.savefig(MODEL_DIR / "ts_decomposition.png", dpi=150,
                    bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close()
        print("Saved: ts_decomposition.png")
    else:
        print("Not enough data for decomposition (need ≥48 obs)")
except ImportError:
    print("statsmodels not installed — skipping decomposition")

# %% ACF / PACF
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=STYLE["bg"])

    plot_acf(hourly.dropna(),  lags=48, ax=axes[0],
             color=STYLE["green"], vlines_kwargs={"colors": STYLE["green"]},
             alpha=0.05)
    axes[0].set_title("Autocorrelation (ACF)")

    plot_pacf(hourly.dropna(), lags=48, ax=axes[1],
              color=STYLE["blue"], vlines_kwargs={"colors": STYLE["blue"]},
              method="ywm", alpha=0.05)
    axes[1].set_title("Partial Autocorrelation (PACF)")

    for ax in axes:
        ax.set_facecolor(STYLE["panel"])
        for line in ax.lines:
            line.set_color(STYLE["muted"] if line.get_color() == "b" else line.get_color())

    plt.tight_layout()
    fig.savefig(MODEL_DIR / "ts_acf_pacf.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("Saved: ts_acf_pacf.png")
except ImportError:
    print("statsmodels not installed — skipping ACF/PACF")

# %% Sequence statistics for LSTM sizing
from config.config import LSTM_PARAMS
SEQ_LEN = LSTM_PARAMS["seq_len"]
data    = hourly.dropna().values
n_seq   = len(data) - SEQ_LEN
print(f"\n── LSTM Sequence Stats ────────────────────")
print(f"  Series length  : {len(data)}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Total sequences: {n_seq}")
print(f"  Train samples  : {int(n_seq * 0.8)}")
print(f"  Val   samples  : {n_seq - int(n_seq * 0.8)}")

# Rolling statistics
fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=STYLE["bg"])

rolling_mean = pd.Series(data).rolling(24).mean()
rolling_std  = pd.Series(data).rolling(24).std()

axes[0].plot(data,         color=STYLE["blue"],   lw=0.8, alpha=0.6, label="Raw")
axes[0].plot(rolling_mean, color=STYLE["green"],  lw=1.5, label="24h Rolling Mean")
axes[0].fill_between(range(len(rolling_mean)),
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     color=STYLE["green"], alpha=0.12)
axes[0].set_title("Rolling Statistics (24h window)")
axes[0].legend(facecolor=STYLE["panel"])

# Lag scatter plot (lag=1)
axes[1].scatter(data[:-1], data[1:], alpha=0.3, s=6, color=STYLE["orange"])
axes[1].set_xlabel("Wait (t)")
axes[1].set_ylabel("Wait (t+1)")
axes[1].set_title("Lag-1 Scatter Plot")

plt.tight_layout()
fig.savefig(MODEL_DIR / "ts_rolling_lag.png", dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("\n✅ Time Series Analysis complete — charts in models/")
