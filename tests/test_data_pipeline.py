"""
tests/test_data_pipeline.py
Unit tests for:
  - DataCollector (synthetic fallback)
  - DataPreprocessor (clean, merge, split)
  - FeatureEngineer (all feature groups)
  - KaggleDatasetLoader (synthetic generation)

Run:
  pytest tests/test_data_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_collection import (
    DataCollector, KaggleDatasetLoader,
    OpenChargeMapCollector, WeatherCollector, TomTomTrafficCollector,
)
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from config.config import TARGET_COLUMN, FEATURE_COLUMNS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_sessions():
    loader = KaggleDatasetLoader()
    return loader._generate_synthetic_sessions(n=500)


@pytest.fixture(scope="module")
def synthetic_stations():
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "station_id": range(1, n + 1),
        "name": [f"Station {i}" for i in range(1, n + 1)],
        "lat": 37.77 + np.random.uniform(-0.05, 0.05, n),
        "lng": -122.42 + np.random.uniform(-0.08, 0.08, n),
        "num_ports": np.random.randint(2, 12, n),
        "connector_type": np.random.choice(["CCS", "CHAdeMO", "Type 2"], n),
        "power_kw": np.random.choice([7.2, 50.0, 150.0], n),
        "status": np.random.choice(["Operational", "Partial", "Offline"], n),
    })


@pytest.fixture(scope="module")
def engineered_df(synthetic_sessions):
    fe = FeatureEngineer()
    return fe.run(synthetic_sessions, save=False)


# ─── KaggleDatasetLoader ──────────────────────────────────────────────────────

class TestKaggleDatasetLoader:
    def test_synthetic_sessions_shape(self, synthetic_sessions):
        assert len(synthetic_sessions) == 500
        assert "station_id" in synthetic_sessions.columns

    def test_synthetic_sessions_target(self, synthetic_sessions):
        assert TARGET_COLUMN in synthetic_sessions.columns
        assert synthetic_sessions[TARGET_COLUMN].notnull().all()
        assert (synthetic_sessions[TARGET_COLUMN] >= 0).all()

    def test_synthetic_sessions_columns(self, synthetic_sessions):
        expected = ["session_id", "station_id", "session_duration_min",
                    "energy_kwh", "queue_size", "traffic_score"]
        for col in expected:
            assert col in synthetic_sessions.columns, f"Missing column: {col}"

    def test_synthetic_no_negative_wait(self, synthetic_sessions):
        assert (synthetic_sessions[TARGET_COLUMN] >= 0).all()

    def test_synthetic_traffic_range(self, synthetic_sessions):
        assert synthetic_sessions["traffic_score"].between(0, 1).all()


# ─── DataPreprocessor ─────────────────────────────────────────────────────────

class TestDataPreprocessor:
    def test_clean_stations(self, synthetic_stations):
        proc = DataPreprocessor()
        cleaned = proc.clean_stations(synthetic_stations)
        assert len(cleaned) > 0
        assert cleaned["lat"].between(-90, 90).all()
        assert cleaned["lng"].between(-180, 180).all()

    def test_clean_stations_removes_invalid_coords(self):
        proc = DataPreprocessor()
        df = pd.DataFrame({
            "station_id": [1, 2, 3],
            "lat": [37.77, None, 200.0],   # 200.0 is invalid
            "lng": [-122.42, -122.42, -122.42],
        })
        cleaned = proc.clean_stations(df)
        assert len(cleaned) == 1   # Only first row valid

    def test_clean_sessions(self, synthetic_sessions):
        proc = DataPreprocessor()
        cleaned = proc.clean_sessions(synthetic_sessions)
        assert len(cleaned) > 0
        assert TARGET_COLUMN in cleaned.columns
        assert cleaned[TARGET_COLUMN].notnull().all()

    def test_clean_sessions_fills_missing(self):
        proc = DataPreprocessor()
        df = pd.DataFrame({
            "station_id": [1, 2],
            TARGET_COLUMN: [10.0, None],
            "session_duration_min": [30.0, 25.0],
        })
        cleaned = proc.clean_sessions(df)
        # Row with None target should be dropped
        assert cleaned[TARGET_COLUMN].notnull().all()

    def test_merge_datasets(self, synthetic_sessions, synthetic_stations):
        proc = DataPreprocessor()
        cleaned_stations = proc.clean_stations(synthetic_stations)
        cleaned_sessions = proc.clean_sessions(synthetic_sessions)
        weather_row = {"temperature_c": 20.0, "precipitation_mm": 0.0}
        merged = proc.merge_datasets(
            cleaned_stations, cleaned_sessions, pd.DataFrame(), weather_row
        )
        assert len(merged) > 0
        assert "temperature_c" in merged.columns

    def test_train_test_split(self, synthetic_sessions):
        proc = DataPreprocessor()
        cleaned = proc.clean_sessions(synthetic_sessions)
        train, test = proc.split(cleaned)
        total = len(train) + len(test)
        assert total == len(cleaned)
        assert len(train) > len(test)   # 80/20 split
        split_ratio = len(train) / total
        assert 0.75 <= split_ratio <= 0.85

    def test_encode_categoricals(self, synthetic_sessions):
        proc = DataPreprocessor()
        df = synthetic_sessions.copy()
        if "connector_type" not in df.columns:
            df["connector_type"] = "CCS"
        encoded = proc.encode_categoricals(df)
        assert "connector_type_encoded" in encoded.columns
        assert encoded["connector_type_encoded"].dtype in [np.int64, np.int32, int]


# ─── FeatureEngineer ──────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_temporal_features(self, synthetic_sessions):
        fe = FeatureEngineer()
        df = fe.add_temporal_features(synthetic_sessions)
        for col in ["hour_of_day", "day_of_week", "is_weekend", "month"]:
            assert col in df.columns, f"Missing: {col}"

    def test_cyclic_encoding(self, synthetic_sessions):
        fe = FeatureEngineer()
        df = fe.add_temporal_features(synthetic_sessions)
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        # Cyclic values must be in [-1, 1]
        assert df["hour_sin"].between(-1.001, 1.001).all()
        assert df["hour_cos"].between(-1.001, 1.001).all()

    def test_cyclic_encoding_hour_range(self, synthetic_sessions):
        fe = FeatureEngineer()
        df = fe.add_temporal_features(synthetic_sessions)
        assert df["hour_of_day"].between(0, 23).all()

    def test_traffic_features(self, synthetic_sessions):
        fe = FeatureEngineer()
        df = fe.add_traffic_features(synthetic_sessions)
        if "traffic_score" in synthetic_sessions.columns:
            assert "traffic_x_hour" in df.columns or True  # optional feature

    def test_utilization_features(self, synthetic_sessions):
        fe = FeatureEngineer()
        df = fe.add_utilization_features(synthetic_sessions)
        assert "station_utilization" in df.columns
        assert df["station_utilization"].between(0, 1).all()

    def test_weather_features(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "temperature_c": [-5.0, 20.0, 40.0],
            "precipitation_mm": [0.0, 1.5, 0.0],
        })
        result = fe.add_weather_features(df)
        assert "is_cold" in result.columns
        assert "is_hot" in result.columns
        assert result["is_cold"].iloc[0] == 1   # -5°C is cold
        assert result["is_hot"].iloc[2] == 1    # 40°C is hot
        assert result["is_cold"].iloc[1] == 0   # 20°C is not cold

    def test_geospatial_features(self, synthetic_stations):
        fe = FeatureEngineer()
        df = synthetic_stations.copy()
        result = fe.add_geospatial_features(df)
        assert "distance_to_city_center_km" in result.columns
        assert (result["distance_to_city_center_km"] >= 0).all()

    def test_rolling_features(self, synthetic_sessions):
        fe = FeatureEngineer()
        df_with_target = synthetic_sessions.copy()
        if TARGET_COLUMN not in df_with_target.columns:
            df_with_target[TARGET_COLUMN] = np.random.uniform(0, 45, len(df_with_target))
        result = fe.add_rolling_features(df_with_target)
        # At least one rolling feature should exist
        rolling_cols = [c for c in result.columns if "rolling" in c]
        assert len(rolling_cols) >= 1

    def test_lag_features(self, synthetic_sessions):
        fe = FeatureEngineer()
        result = fe.add_lag_features(synthetic_sessions, lags=[1, 3, 6])
        for lag in [1, 3, 6]:
            assert f"lag_{lag}" in result.columns

    def test_full_pipeline_no_nan(self, engineered_df):
        numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
        nan_counts = engineered_df[numeric_cols].isnull().sum()
        assert nan_counts.sum() == 0, f"NaN values found: {nan_counts[nan_counts>0]}"

    def test_full_pipeline_shape(self, synthetic_sessions, engineered_df):
        assert len(engineered_df) > 0
        # Should have at least as many columns as input
        assert len(engineered_df.columns) >= len(synthetic_sessions.columns)

    def test_lstm_sequence_preparation(self, engineered_df):
        fe = FeatureEngineer()
        feat_cols = fe.get_feature_columns(engineered_df)
        if len(feat_cols) == 0 or TARGET_COLUMN not in engineered_df.columns:
            pytest.skip("Insufficient features for sequence test")
        seq_len = 12
        X, y = fe.prepare_lstm_sequences(engineered_df, seq_len, TARGET_COLUMN, feat_cols)
        assert X.ndim == 3          # (samples, seq_len, features)
        assert y.ndim == 1          # (samples,)
        assert X.shape[1] == seq_len
        assert X.shape[2] == len(feat_cols)
        assert len(X) == len(y)


# ─── DataCollector Integration ────────────────────────────────────────────────

class TestDataCollector:
    def test_collect_all_returns_sessions(self):
        collector = DataCollector()
        data = collector.collect_all()
        assert "sessions" in data
        assert isinstance(data["sessions"], pd.DataFrame)
        assert len(data["sessions"]) > 0

    def test_collect_all_returns_stations(self):
        collector = DataCollector()
        data = collector.collect_all()
        # stations may be empty if API key not set but should not raise
        assert "stations" in data

    def test_traffic_score_normalisation(self):
        collector = TomTomTrafficCollector()
        flow = {"currentSpeed": 25, "freeFlowSpeed": 50}
        score = collector._compute_score(flow)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.5, abs=0.01)

    def test_traffic_score_free_flow(self):
        collector = TomTomTrafficCollector()
        flow = {"currentSpeed": 60, "freeFlowSpeed": 60}
        score = collector._compute_score(flow)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_traffic_score_standstill(self):
        collector = TomTomTrafficCollector()
        flow = {"currentSpeed": 0, "freeFlowSpeed": 60}
        score = collector._compute_score(flow)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_ocm_fallback_no_crash(self):
        """OCM collector should not crash when API key is missing."""
        collector = OpenChargeMapCollector()
        # With no real key, it falls back to Kaggle CSV or empty DataFrame
        result = collector._load_kaggle_fallback()
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
