"""
src/feature_engineering.py
Creates all model-ready features from cleaned data:
  - Temporal features (cyclic encoding of hour/day/month)
  - Rolling statistics (1h, 3h, 6h windows)
  - Traffic-interaction features
  - Geospatial features (distance to city centre, POI density)
  - Weather interaction features
  - Lag features for time-series models
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    CYCLIC_FEATURES, DEFAULT_LOCATION, FEATURE_COLUMNS,
    PROCESSED_FEATURES_CSV, TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms cleaned data into rich feature set for ML models."""

    def __init__(self, city_centre: dict = DEFAULT_LOCATION):
        self.city_centre = city_centre

    # ─── Temporal ─────────────────────────────────────────────────────────────

    def add_temporal_features(self, df: pd.DataFrame, dt_col: str = "session_start") -> pd.DataFrame:
        """Extract and cyclically encode all time-based features."""
        df = df.copy()

        # Parse datetime if needed
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            dt = df[dt_col]
        else:
            logger.warning(f"Datetime column '{dt_col}' not found — using current time proxy")
            dt = pd.Series(pd.Timestamp.now(), index=df.index)

        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek       # 0=Mon … 6=Sun
        df["month"] = dt.dt.month
        df["day_of_month"] = dt.dt.day
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour_of_day"].apply(
            lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0
        )
        df["part_of_day"] = df["hour_of_day"].apply(self._part_of_day)

        # Cyclic encoding (prevents discontinuity at midnight / Sunday)
        for col, period in CYCLIC_FEATURES.items():
            if col in df.columns:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

        return df

    @staticmethod
    def _part_of_day(hour: int) -> int:
        """0=night, 1=morning, 2=afternoon, 3=evening."""
        if 5 <= hour < 12:
            return 1
        elif 12 <= hour < 17:
            return 2
        elif 17 <= hour < 21:
            return 3
        return 0

    # ─── Rolling Statistics ───────────────────────────────────────────────────

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        group_col: str = "station_id",
        target_col: str = TARGET_COLUMN,
    ) -> pd.DataFrame:
        """Compute rolling mean/std over past 1h, 3h, 6h windows."""
        df = df.copy()
        if target_col not in df.columns:
            logger.warning(f"Target '{target_col}' not found for rolling features")
            return df

        dt_col = "session_start" if "session_start" in df.columns else None
        if dt_col and group_col in df.columns:
            df = df.sort_values([group_col, dt_col])
            for window in [6, 18, 36]:   # rows (assuming ~10 min avg session gap)
                label = {6: "1h", 18: "3h", 36: "6h"}[window]
                df[f"rolling_mean_{label}"] = (
                    df.groupby(group_col)[target_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df[f"rolling_std_{label}"] = (
                    df.groupby(group_col)[target_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
                )
        else:
            for label in ["1h", "3h", "6h"]:
                df[f"rolling_mean_{label}"] = df[target_col].expanding().mean()
                df[f"rolling_std_{label}"] = df[target_col].expanding().std().fillna(0)

        return df

    # ─── Lag Features (for LSTM) ──────────────────────────────────────────────

    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        lags: List[int] = [1, 2, 3, 6, 12, 24],
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        df = df.copy()
        if target_col not in df.columns:
            return df
        for lag in lags:
            col_name = f"lag_{lag}"
            if group_col in df.columns:
                df[col_name] = df.groupby(group_col)[target_col].shift(lag)
            else:
                df[col_name] = df[target_col].shift(lag)
        return df

    # ─── Traffic Interactions ─────────────────────────────────────────────────

    def add_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "traffic_score" in df.columns:
            df["traffic_x_hour"] = df["traffic_score"] * df.get("hour_of_day", 12) / 24
            df["traffic_x_weekend"] = df["traffic_score"] * df.get("is_weekend", 0)
            df["traffic_level"] = pd.cut(
                df["traffic_score"],
                bins=[0, 0.33, 0.66, 1.0],
                labels=[0, 1, 2],   # low / medium / high
                include_lowest=True,
            ).astype(float)
        return df

    # ─── Utilization Features ─────────────────────────────────────────────────

    def add_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        #df = df.copy()
        #avail = df.get("available_ports", pd.Series(2, index=df.index))
        #total = df.get("num_ports", pd.Series(4, index=df.index)).replace(0, 1)
        #df["station_utilization"] = ((total - avail) / total).clip(0, 1)
        #df["port_availability_ratio"] = (avail / total).clip(0, 1)

        #queue = df.get("queue_size", pd.Series(0, index=df.index))
        #df["queue_pressure"] = queue / (avail + 1)   # avoids div-by-zero
        avail = df.get("available_ports", pd.Series(2, index=df.index))
        total = df.get("num_ports", pd.Series(4, index=df.index)).replace(0, 1)

# Prevent invalid values
        avail = avail.clip(lower=0)
        avail = np.minimum(avail, total)

        df["available_ports"] = avail
        df["num_ports"] = total

# Correct utilization calculation
        df["station_utilization"] = ((total - avail) / total).clip(0, 1)

        df["utilization_pct"] = df["station_utilization"] * 100

        df["port_availability_ratio"] = (avail / total).clip(0, 1)

        queue = df.get("queue_size", pd.Series(0, index=df.index))

        df["queue_pressure"] = queue / (avail + 1)
        return df

    # ─── Geospatial ───────────────────────────────────────────────────────────

    def add_geospatial_features(
        self, df: pd.DataFrame, stations_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df = df.copy()
        if "lat" in df.columns and "lng" in df.columns:
            df["distance_to_city_center_km"] = self._haversine(
                df["lat"], df["lng"],
                self.city_centre["lat"], self.city_centre["lng"],
            )
        else:
            df["distance_to_city_center_km"] = 5.0   # default

        # Count nearby stations (within 2 km) if station coordinates available
        if stations_df is not None and not stations_df.empty and "lat" in stations_df.columns:
            df["nearby_stations_count"] = df.apply(
                lambda row: self._count_nearby(row, stations_df, radius_km=2.0)
                if pd.notnull(row.get("lat")) else 3,
                axis=1,
            )
        else:
            df["nearby_stations_count"] = 3   # default

        return df

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> pd.Series:
        R = 6371.0
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _count_nearby(row, stations_df: pd.DataFrame, radius_km: float) -> int:
        dists = FeatureEngineer._haversine(
            row["lat"], row["lng"], stations_df["lat"], stations_df["lng"]
        )
        return int((dists < radius_km).sum())

    # ─── Weather Interactions ─────────────────────────────────────────────────

    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        temp = df.get("temperature_c", pd.Series(20.0, index=df.index))
        precip = df.get("precipitation_mm", pd.Series(0.0, index=df.index))

        df["is_cold"] = (temp < 5).astype(int)
        df["is_hot"] = (temp > 35).astype(int)
        df["is_raining"] = (precip > 0.1).astype(int)
        df["weather_demand_boost"] = df["is_raining"] * 0.15 + df["is_cold"] * 0.10
        return df

    # ─── Master Pipeline ──────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        stations_df: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on {len(df)} rows")

        df = self.add_temporal_features(df)
        df = self.add_traffic_features(df)
        df = self.add_utilization_features(df)
        df = self.add_weather_features(df)
        df = self.add_geospatial_features(df, stations_df)
        df = self.add_rolling_features(df)
        df = self.add_lag_features(df)

        # Drop raw datetime to avoid leakage
        df.drop(columns=["session_start", "datetime", "date_created"],
                errors="ignore", inplace=True)

        # Fill any remaining NaNs with column median
        #numeric_cols = df.select_dtypes(include=[np.number]).columns
        #df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill missing numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        for col in numeric_cols:
            median_val = df[col].median()

            if pd.isna(median_val):
                median_val = 0

            df[col] = df[col].fillna(median_val)
            
        # Final safety corrections
            if "available_ports" in df.columns and "num_ports" in df.columns:
                df["available_ports"] = df["available_ports"].clip(lower=0)

                df["available_ports"] = np.minimum(
                    df["available_ports"],
                    df["num_ports"]
                )

                df["station_utilization"] = (
                    (df["num_ports"] - df["available_ports"])
                    / df["num_ports"]
                ).clip(0, 1)

                df["utilization_pct"] = (
                    df["station_utilization"] * 100
    )
# Remove unwanted columns
        drop_cols = [
            "timestamp",
            "session_id",
            "connector_type_x",
            "connector_type_y",
            "connector_type",
            "weather_main"
        ]

        df.drop(columns=drop_cols, errors="ignore", inplace=True)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        logger.info(f"Feature columns: {list(df.columns)}")

        if save:
            df.to_csv(PROCESSED_FEATURES_CSV, index=False)
            logger.info(f"Saved to {PROCESSED_FEATURES_CSV}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return columns that are actually present in the engineered df."""
        return [c for c in FEATURE_COLUMNS if c in df.columns]

    def prepare_lstm_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 24,
        target_col: str = TARGET_COLUMN,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        Build (X, y) arrays for LSTM training.
        Returns: X shape (samples, seq_len, n_features), y shape (samples,)
        """
        import numpy as np
        if feature_cols is None:
            feature_cols = self.get_feature_columns(df)

        avail_feats = [c for c in feature_cols if c in df.columns]
        X_data = df[avail_feats].values
        y_data = df[target_col].values

        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_data)):
            X_seq.append(X_data[i - sequence_length: i])
            y_seq.append(y_data[i])

        return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # Quick test with synthetic data
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_collection import DataCollector

    collector = DataCollector()
    raw = collector.collect_all()
    sessions = raw.get("sessions", pd.DataFrame())

    if not sessions.empty:
        fe = FeatureEngineer()
        engineered = fe.run(sessions, stations_df=raw.get("stations"))
        print(f"Engineered shape: {engineered.shape}")
        print(engineered.head(3))
