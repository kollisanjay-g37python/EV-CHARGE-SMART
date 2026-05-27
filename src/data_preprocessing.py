"""
src/data_preprocessing.py
Cleans, merges, and validates all raw data into analysis-ready DataFrames.
Steps:
  1. Load raw CSVs
  2. Handle missing values & outliers
  3. Type conversion & datetime parsing
  4. Merge stations + sessions + traffic + weather
  5. Train/test split
  6. Save processed outputs
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    LABEL_ENCODER_PATH, PROCESSED_DIR, PROCESSED_FEATURES_CSV,
    PROCESSED_SESSIONS_CSV, PROCESSED_STATIONS_CSV, RAW_DIR,
    RANDOM_STATE, SCALER_PATH, TARGET_COLUMN, TEST_CSV, TRAIN_CSV,
    TRAIN_TEST_SPLIT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Complete preprocessing pipeline for EV charging data."""

    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ─── Loaders ──────────────────────────────────────────────────────────────

    def load_raw(self) -> dict:
        raw = {}
        for name in ["stations_raw", "sessions_raw", "traffic_raw", "weather_raw",
                     "energy_raw", "demand_raw"]:
            path = RAW_DIR / f"{name}.csv"
            if path.exists():
                raw[name] = pd.read_csv(path)
                logger.info(f"Loaded {name}: {raw[name].shape}")
            else:
                logger.warning(f"Missing raw file: {path}")
        return raw

    # ─── Station Cleaning ──────────────────────────────────────────────────────

    def clean_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning station data...")
        df = df.copy()

        # Drop rows missing critical location
        df.dropna(subset=["lat", "lng"], inplace=True)

        # Clip coordinates to valid range
        df = df[(df["lat"].between(-90, 90)) & (df["lng"].between(-180, 180))]

        # Fill missing values
        df["num_ports"] = df.get("num_ports", pd.Series(dtype=int)).fillna(2).astype(int)
        df["connector_type"] = df.get("connector_type", pd.Series(dtype=str)).fillna("Unknown")
        df["power_kw"] = df.get("power_kw", pd.Series(dtype=float)).fillna(50.0)
        df["status"] = df.get("status", pd.Series(dtype=str)).fillna("Unknown")

        # Encode connector types
        known_connectors = ["Type 2", "CCS", "CHAdeMO", "Tesla", "J1772", "Unknown"]
        df["connector_type"] = df["connector_type"].apply(
            lambda x: x if any(k in str(x) for k in known_connectors) else "Unknown"
        )

        # Create binary status flag
        df["is_operational"] = df["status"].str.contains(
            "Operational|Available|Online", case=False, na=False
        ).astype(int)

        logger.info(f"Stations after cleaning: {len(df)}")
        return df.reset_index(drop=True)

    # ─── Session Cleaning ──────────────────────────────────────────────────────

    def clean_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning session data...")
        df = df.copy()

        # Parse datetime
        for col in ["start_time", "end_time", "datetime", "Datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Standardise column names
        rename_map = {
            "start_time": "session_start",
            "session_duration_min": "session_duration_min",
            "energy_kwh": "energy_kwh",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        if "session_start" not in df.columns and "datetime" in df.columns:
            df["session_start"] = df["datetime"]

        # Remove extreme outliers using IQR
        for col in ["session_duration_min", "energy_kwh", "wait_time_minutes"]:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.01, 0.99])
                df = df[df[col].between(q1, q3)]

        # Ensure target column exists
        if TARGET_COLUMN not in df.columns:
            logger.warning(f"Target '{TARGET_COLUMN}' not in data — generating proxy")
            df[TARGET_COLUMN] = self._compute_wait_time(df)

        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        logger.info(f"Sessions after cleaning: {len(df)}")
        return df.reset_index(drop=True)

    def _compute_wait_time(self, df: pd.DataFrame) -> pd.Series:
        """Heuristic wait time from queue and port availability."""
        queue = df.get("queue_size", pd.Series(0, index=df.index)).fillna(0)
        avail = df.get("available_ports", pd.Series(2, index=df.index)).fillna(2)
        dur = df.get("session_duration_min", pd.Series(30, index=df.index)).fillna(30)
        traffic = df.get("traffic_score", pd.Series(0.3, index=df.index)).fillna(0.3)
        wait = (queue / (avail + 1)) * dur + traffic * 5
        return wait.clip(0, 120).round(1)

    # ─── Traffic Cleaning ─────────────────────────────────────────────────────

    def clean_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["traffic_score"] = df.get("traffic_score", pd.Series(dtype=float)).fillna(0.3).clip(0, 1)
        return df

    # ─── Merge ────────────────────────────────────────────────────────────────

    def merge_datasets(
        self,
        stations: pd.DataFrame,
        sessions: pd.DataFrame,
        traffic: pd.DataFrame,
        weather_row: dict,
    ) -> pd.DataFrame:
        logger.info("Merging datasets...")
        df = sessions.copy()

        # Merge station metadata if station_id present in both
        if "station_id" in df.columns and not stations.empty:
            station_cols = ["station_id", "lat", "lng", "num_ports",
                            "connector_type", "power_kw", "is_operational"]
            avail_cols = [c for c in station_cols if c in stations.columns]
            df = df.merge(stations[avail_cols], on="station_id", how="left")

        # Merge traffic
        if not traffic.empty and "station_id" in traffic.columns:
            df = df.merge(
                traffic[["station_id", "traffic_score"]],
                on="station_id", how="left", suffixes=("", "_traffic"),
            )
            # Prefer session-level traffic, fallback to station-level
            if "traffic_score_traffic" in df.columns:
                df["traffic_score"] = df["traffic_score"].fillna(df["traffic_score_traffic"])
                df.drop(columns=["traffic_score_traffic"], inplace=True)

        # Append weather as constants (same collection point for demo)
        for k, v in weather_row.items():
            if k not in df.columns:
                df[k] = v

        # Fill remaining nulls
        df["traffic_score"] = df.get("traffic_score", pd.Series(dtype=float)).fillna(0.3)
        df["temperature_c"] = df.get("temperature_c", pd.Series(dtype=float)).fillna(20.0)
        df["precipitation_mm"] = df.get("precipitation_mm", pd.Series(dtype=float)).fillna(0.0)
        df["connector_type"] = df.get("connector_type", pd.Series(dtype=str)).fillna("Unknown")

        logger.info(f"Merged dataset shape: {df.shape}")
        return df

    # ─── Encode & Scale ───────────────────────────────────────────────────────

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "connector_type" in df.columns:
            df["connector_type_encoded"] = self.label_encoder.fit_transform(
                df["connector_type"].astype(str)
            )
            joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
            logger.info(f"Label encoder saved to {LABEL_ENCODER_PATH}")
        return df

    def scale_features(
        self, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        avail = [c for c in feature_cols if c in train.columns]
        train[avail] = self.scaler.fit_transform(train[avail])
        test[avail] = self.scaler.transform(test[avail])
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info(f"Scaler saved to {SCALER_PATH}")
        return train, test

    # ─── Train/Test Split ─────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(
            df, test_size=1 - TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True
        )
        logger.info(f"Train: {len(train)} | Test: {len(test)}")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    # ─── Master Pipeline ──────────────────────────────────────────────────────

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full preprocessing pipeline. Returns (train_df, test_df)."""
        raw = self.load_raw()

        stations = self.clean_stations(raw.get("stations_raw", pd.DataFrame()))
        sessions = self.clean_sessions(raw.get("sessions_raw", pd.DataFrame()))
        traffic = self.clean_traffic(raw.get("traffic_raw", pd.DataFrame()))
        weather_row = (
            raw["weather_raw"].iloc[0].to_dict()
            if "weather_raw" in raw and not raw["weather_raw"].empty
            else {"temperature_c": 20.0, "precipitation_mm": 0.0}
        )

        stations.to_csv(PROCESSED_STATIONS_CSV, index=False)
        sessions.to_csv(PROCESSED_SESSIONS_CSV, index=False)

        merged = self.merge_datasets(stations, sessions, traffic, weather_row)
        merged = self.encode_categoricals(merged)

        train, test = self.split(merged)
        train.to_csv(TRAIN_CSV, index=False)
        test.to_csv(TEST_CSV, index=False)
        merged.to_csv(PROCESSED_FEATURES_CSV, index=False)

        logger.info("=== Preprocessing complete ===")
        return train, test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    train, test = preprocessor.run()
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"Columns: {list(train.columns)}")
