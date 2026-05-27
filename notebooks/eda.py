# notebooks/eda.py  (run as: jupyter nbconvert --to notebook --execute)
# ─── EV Charging Dataset – Exploratory Data Analysis ─────────────────────────

# %% [markdown]
# # ⚡ EV Charging Smart System — EDA
# Explores station metadata, session patterns, and feature distributions.

# %% Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from config.config import RAW_DIR, PROC_DIR, MODEL_DIR

STYLE = dict(bg="#0d1117", panel="#161b22", green="#39d353",
             blue="#58a6ff", orange="#f78166", text="#c9d1d9")
plt.rcParams.update({
    "axes.facecolor":  STYLE["panel"],
    "figure.facecolor":STYLE["bg"],
    "text.color":      STYLE["text"],
    "axes.labelcolor": STYLE["text"],
    "xtick.color":     STYLE["text"],
    "ytick.color":     STYLE["text"],
    "axes.edgecolor":  "#21262d",
    "grid.color":      "#21262d",
    "axes.grid":       True,
})

# %% Load data
sessions = pd.read_csv(RAW_DIR / "charging_sessions.csv", parse_dates=["start_time"])
stations = pd.read_csv(RAW_DIR / "stations.csv")
print(f"Sessions : {len(sessions):,}  cols={list(sessions.columns)}")
print(f"Stations : {len(stations):,}  cols={list(stations.columns)}")
print(sessions.describe())

# %% Wait time distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=STYLE["bg"])

axes[0].hist(sessions["wait_time_minutes"], bins=50,
             color=STYLE["green"], alpha=0.85, edgecolor="none")
axes[0].set_xlabel("Wait Time (min)")
axes[0].set_ylabel("Count")
axes[0].set_title("Wait Time Distribution")

axes[1].hist(sessions["duration_minutes"], bins=50,
             color=STYLE["blue"], alpha=0.85, edgecolor="none")
axes[1].set_xlabel("Charging Duration (min)")
axes[1].set_title("Session Duration Distribution")

axes[2].hist(sessions["energy_kwh"], bins=50,
             color=STYLE["orange"], alpha=0.85, edgecolor="none")
axes[2].set_xlabel("Energy (kWh)")
axes[2].set_title("Energy per Session")

plt.tight_layout()
fig.savefig(MODEL_DIR / "eda_distributions.png", dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Saved: eda_distributions.png")

# %% Temporal patterns
sessions["hour"]       = sessions["start_time"].dt.hour
sessions["day_of_week"]= sessions["start_time"].dt.dayofweek
sessions["month"]      = sessions["start_time"].dt.month
sessions["is_weekend"] = sessions["day_of_week"] >= 5

fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=STYLE["bg"])

# Hourly sessions
hourly = sessions.groupby("hour")["wait_time_minutes"].agg(["mean","std"])
axes[0,0].plot(hourly.index, hourly["mean"], color=STYLE["green"], lw=2.5)
axes[0,0].fill_between(hourly.index,
    hourly["mean"] - hourly["std"],
    hourly["mean"] + hourly["std"],
    color=STYLE["green"], alpha=0.15)
axes[0,0].set_xlabel("Hour of Day")
axes[0,0].set_ylabel("Avg Wait (min)")
axes[0,0].set_title("Wait Time by Hour (±1 std)")
axes[0,0].axvspan(7, 9,  alpha=0.12, color=STYLE["orange"], label="AM Peak")
axes[0,0].axvspan(17, 19, alpha=0.12, color=STYLE["blue"],   label="PM Peak")
axes[0,0].legend(facecolor=STYLE["panel"])

# DOW pattern
dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow = sessions.groupby("day_of_week")["wait_time_minutes"].mean()
axes[0,1].bar(dow.index, dow.values,
              color=[STYLE["orange"] if i >= 5 else STYLE["blue"] for i in dow.index],
              edgecolor="none")
axes[0,1].set_xticks(range(7))
axes[0,1].set_xticklabels(dow_names)
axes[0,1].set_ylabel("Avg Wait (min)")
axes[0,1].set_title("Wait Time by Day of Week")

# Monthly trend
monthly = sessions.groupby("month")["wait_time_minutes"].mean()
axes[1,0].plot(monthly.index, monthly.values,
               color=STYLE["blue"], lw=2.5, marker="o", markersize=6)
axes[1,0].set_xlabel("Month")
axes[1,0].set_ylabel("Avg Wait (min)")
axes[1,0].set_title("Monthly Wait Trend")

# Weekend vs weekday
wd = sessions.groupby(["is_weekend","hour"])["wait_time_minutes"].mean().unstack(0)
axes[1,1].plot(wd.index, wd[False], label="Weekday", color=STYLE["blue"], lw=2)
axes[1,1].plot(wd.index, wd[True],  label="Weekend", color=STYLE["orange"], lw=2)
axes[1,1].set_xlabel("Hour")
axes[1,1].set_ylabel("Avg Wait (min)")
axes[1,1].set_title("Weekday vs Weekend Pattern")
axes[1,1].legend(facecolor=STYLE["panel"])

plt.tight_layout()
fig.savefig(MODEL_DIR / "eda_temporal.png", dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Saved: eda_temporal.png")

# %% Station analysis
station_stats = (sessions.groupby("station_id")
                  .agg(avg_wait=("wait_time_minutes","mean"),
                       total_sessions=("wait_time_minutes","count"),
                       avg_duration=("duration_minutes","mean"))
                  .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=STYLE["bg"])

axes[0].scatter(station_stats["total_sessions"],
                station_stats["avg_wait"],
                alpha=0.5, s=20, color=STYLE["green"])
axes[0].set_xlabel("Total Sessions")
axes[0].set_ylabel("Avg Wait (min)")
axes[0].set_title("Station Load vs Average Wait")

axes[1].hist(station_stats["avg_wait"], bins=30,
             color=STYLE["orange"], alpha=0.85, edgecolor="none")
axes[1].set_xlabel("Station Average Wait (min)")
axes[1].set_ylabel("Number of Stations")
axes[1].set_title("Distribution of Station-Level Average Wait")

plt.tight_layout()
fig.savefig(MODEL_DIR / "eda_stations.png", dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Saved: eda_stations.png")

# %% Correlation heatmap
try:
    feat_df = pd.read_csv(PROC_DIR / "features.csv")
    num_cols = feat_df.select_dtypes(include=np.number).columns[:12]
    corr = feat_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=STYLE["bg"])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax,
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3, linecolor=STYLE["bg"],
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=STYLE["text"], pad=12)
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "eda_correlation.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("Saved: eda_correlation.png")
except FileNotFoundError:
    print("features.csv not found — run feature_engineering.py first")

print("\n✅ EDA complete — all charts in models/")
