"""
src/utils.py
Shared utility functions used across the EV Charging Smart System.
"""

import logging
import hashlib
import json
import time
import math
import functools
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure consistent logging for all modules."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ─── Timing decorator ─────────────────────────────────────────────────────────

def timed(fn):
    """Decorator that logs function execution time."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.debug(f"{fn.__qualname__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ─── Geospatial ───────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bounding_box(lat: float, lon: float, radius_km: float) -> dict:
    """Return lat/lon bounding box for a circle of given radius."""
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return {
        "lat_min": lat - delta_lat,
        "lat_max": lat + delta_lat,
        "lon_min": lon - delta_lon,
        "lon_max": lon + delta_lon,
    }


def filter_by_radius(df: pd.DataFrame,
                      center_lat: float, center_lon: float,
                      radius_km: float) -> pd.DataFrame:
    """Filter station DataFrame to those within radius_km of centre."""
    df = df.copy()
    df["_dist_km"] = df.apply(
        lambda r: haversine_km(center_lat, center_lon,
                               float(r["lat"]), float(r["lon"])), axis=1
    )
    return (df[df["_dist_km"] <= radius_km]
              .sort_values("_dist_km")
              .drop(columns=["_dist_km"])
              .reset_index(drop=True))


# ─── Availability helpers ────────────────────────────────────────────────────

def availability_label(utilisation: float) -> str:
    """Map utilisation ratio [0, ∞) to a human-readable label."""
    if utilisation < 0.4:  return "Available"
    if utilisation < 0.7:  return "Moderate"
    if utilisation < 0.9:  return "Busy"
    return "Full"


def availability_color(label: str) -> str:
    """Return hex colour for availability label."""
    return {
        "Available": "#39d353",
        "Moderate":  "#e3b341",
        "Busy":      "#f78166",
        "Full":      "#da3633",
    }.get(label, "#8b949e")


# ─── Data validation ─────────────────────────────────────────────────────────

def validate_station_row(row: dict) -> bool:
    """Return True if a station dict has the minimum required fields."""
    required = {"station_id", "lat", "lon", "num_connectors"}
    return required.issubset(set(row.keys()))


def safe_float(val: Any, default: float = 0.0) -> float:
    """Convert value to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ─── Cache helpers ────────────────────────────────────────────────────────────

def cache_key(*args) -> str:
    """Create a short deterministic cache key from arguments."""
    raw = "_".join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class SimpleFileCache:
    """
    Lightweight file-based key-value cache with TTL.
    Values are serialised as JSON.
    """
    def __init__(self, cache_dir: Path, ttl_seconds: int = 300):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl:
            p.unlink(missing_ok=True)
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def set(self, key: str, value: Any):
        try:
            self._path(key).write_text(json.dumps(value))
        except Exception as e:
            log.warning(f"Cache write failed for key={key}: {e}")

    def invalidate(self, key: str):
        self._path(key).unlink(missing_ok=True)

    def clear(self):
        for f in self.dir.glob("*.json"):
            f.unlink(missing_ok=True)


# ─── DataFrame helpers ───────────────────────────────────────────────────────

def ensure_columns(df: pd.DataFrame, cols: list[str],
                   default: float = 0.0) -> pd.DataFrame:
    """Add missing columns to a DataFrame with a default value."""
    for col in cols:
        if col not in df.columns:
            df[col] = default
    return df


def clip_column(df: pd.DataFrame, col: str,
                lo: float = 0.0, hi: float = 120.0) -> pd.DataFrame:
    """Clip a numeric column in-place and return the DataFrame."""
    if col in df.columns:
        df[col] = df[col].clip(lo, hi)
    return df


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return memory usage of a DataFrame in MB."""
    return round(df.memory_usage(deep=True).sum() / 1_048_576, 2)


# ─── Formatting ──────────────────────────────────────────────────────────────

def fmt_wait(minutes: float) -> str:
    """Human-readable wait time string."""
    if minutes < 1:   return "< 1 min"
    if minutes < 60:  return f"{int(round(minutes))} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m"


def fmt_distance(km: float) -> str:
    """Human-readable distance string."""
    if km < 1.0:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"


def timestamp_now() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Numeric helpers ─────────────────────────────────────────────────────────

def running_average(new_val: float, prev_avg: float, n: int) -> float:
    """Online/incremental mean update."""
    return (prev_avg * (n - 1) + new_val) / n


def normalise_0_1(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


if __name__ == "__main__":
    # Quick smoke tests
    print(f"SF → Oakland : {haversine_km(37.7749,-122.4194,37.8044,-122.2712):.2f} km")
    print(f"fmt_wait(75) : {fmt_wait(75)}")
    print(f"fmt_distance(0.45) : {fmt_distance(0.45)}")
    print(f"availability_label(0.65) : {availability_label(0.65)}")
