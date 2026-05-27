# tests/test_pipeline.py
"""
Unit & integration tests for the EV Charging Smart System pipeline.
Run: pytest tests/ -v --tb=short
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_stations():
    """20 synthetic stations."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "station_id":     range(1, 21),
        "name":           [f"Station {i}" for i in range(1, 21)],
        "lat":            37.77 + rng.uniform(-0.1, 0.1, 20),
        "lon":            -122.41 + rng.uniform(-0.1, 0.1, 20),
        "num_connectors": rng.integers(2, 12, 20),
        "connector_types":["CCS,CHAdeMO"] * 20,
        "status":         ["Operational"] * 20,
        "is_free":        rng.choice([True, False], 20),
        "operator":       ["ChargePoint"] * 20,
    })


@pytest.fixture(scope="session")
def sample_sessions(sample_stations):
    """500 synthetic sessions for fast tests."""
    rng = np.random.default_rng(0)
    n   = 500
    return pd.DataFrame({
        "station_id":        rng.integers(1, 21, n),
        "start_time":        pd.date_range("2024-01-01", periods=n, freq="1h"),
        "duration_minutes":  rng.integers(10, 120, n).astype(float),
        "wait_time_minutes": rng.integers(0, 40, n).astype(float),
        "energy_kwh":        rng.uniform(5, 60, n),
        "connector_type":    "CCS",
        "lat":               37.77 + rng.uniform(-0.1, 0.1, n),
        "lon":               -122.41 + rng.uniform(-0.1, 0.1, n),
    })


# ─── Data Collection ──────────────────────────────────────────────────────────

class TestDataCollection:

    def test_synthetic_stations_shape(self):
        from src.data_collection import _synthetic_stations
        df = _synthetic_stations(n=50)
        assert len(df) == 50
        assert "station_id" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns

    def test_synthetic_stations_valid_coords(self):
        from src.data_collection import _synthetic_stations
        df = _synthetic_stations(n=30)
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(-180, 180).all()

    def test_synthetic_sessions_columns(self, sample_stations):
        from src.data_collection import generate_synthetic_sessions
        df = generate_synthetic_sessions(sample_stations, days=3)
        required = ["station_id","start_time","wait_time_minutes","duration_minutes"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_synthetic_sessions_wait_clipped(self, sample_stations):
        from src.data_collection import generate_synthetic_sessions
        df = generate_synthetic_sessions(sample_stations, days=3)
        assert df["wait_time_minutes"].min() >= 0

    def test_traffic_index_range(self):
        from src.data_collection import fetch_traffic_index
        idx = fetch_traffic_index(37.7749, -122.4194)
        assert 0.0 <= idx <= 1.0

    def test_weather_fallback(self):
        from src.data_collection import fetch_weather
        w = fetch_weather(37.7749, -122.4194)
        assert "temperature"   in w
        assert "precipitation" in w
        assert isinstance(w["temperature"], float)


# ─── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_clean_removes_negatives(self, sample_sessions):
        from src.data_preprocessing import clean_sessions
        dirty = sample_sessions.copy()
        dirty.loc[0, "wait_time_minutes"] = -5
        dirty.loc[1, "wait_time_minutes"] = 200   # above 120 cap
        cleaned = clean_sessions(dirty)
        assert (cleaned["wait_time_minutes"] >= 0).all()
        assert (cleaned["wait_time_minutes"] <= 120).all()

    def test_clean_drops_na_target(self, sample_sessions):
        from src.data_preprocessing import clean_sessions
        dirty = sample_sessions.copy()
        dirty.loc[0, "wait_time_minutes"] = float("nan")
        cleaned = clean_sessions(dirty)
        assert cleaned["wait_time_minutes"].isna().sum() == 0

    def test_encode_categoricals(self, sample_sessions):
        from src.data_preprocessing import encode_categoricals
        df = sample_sessions.copy()
        df["connector_type"] = "CCS"
        df["operator"]       = "ChargePoint"
        out = encode_categoricals(df)
        assert "connector_type_enc" in out.columns
        assert out["connector_type_enc"].dtype in [int, np.int64, np.int32]

    def test_merge_station_meta(self, sample_sessions, sample_stations):
        from src.data_preprocessing import merge_station_meta
        merged = merge_station_meta(sample_sessions, sample_stations)
        assert "num_connectors" in merged.columns


# ─── Feature Engineering ─────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_temporal_features(self, sample_sessions):
        from src.feature_engineering import add_temporal_features
        df  = sample_sessions.copy()
        out = add_temporal_features(df)
        for col in ["hour","day_of_week","month","is_weekend","is_peak_hour"]:
            assert col in out.columns, f"Missing: {col}"

    def test_hour_range(self, sample_sessions):
        from src.feature_engineering import add_temporal_features
        out = add_temporal_features(sample_sessions.copy())
        assert out["hour"].between(0, 23).all()

    def test_cyclical_encoding(self, sample_sessions):
        from src.feature_engineering import add_temporal_features
        out = add_temporal_features(sample_sessions.copy())
        assert out["hour_sin"].between(-1, 1).all()
        assert out["hour_cos"].between(-1, 1).all()

    def test_traffic_proxy_range(self, sample_sessions):
        from src.feature_engineering import add_temporal_features, add_traffic_proxy
        df  = add_temporal_features(sample_sessions.copy())
        out = add_traffic_proxy(df)
        assert out["traffic_index"].between(0, 1).all()

    def test_rolling_features(self, sample_sessions):
        from src.feature_engineering import add_rolling_features
        df = sample_sessions.copy()
        df["wait_time_minutes"] = df["wait_time_minutes"].astype(float)
        out = add_rolling_features(df)
        assert "rolling_mean_1h" in out.columns
        assert "rolling_mean_3h" in out.columns

    def test_interaction_features(self, sample_sessions):
        from src.feature_engineering import (add_temporal_features, add_traffic_proxy,
                                              add_station_features, add_interaction_features)
        df = add_temporal_features(sample_sessions.copy())
        df = add_traffic_proxy(df)
        df = add_station_features(df)
        out = add_interaction_features(df)
        assert "traffic_x_peak"     in out.columns
        assert "capacity_x_traffic" in out.columns


# ─── Queue Model ─────────────────────────────────────────────────────────────

class TestQueueModel:

    def test_erlang_c_stable(self):
        from src.queue_model import erlang_c
        # rho < 1 → valid probability
        p = erlang_c(c=4, rho=0.5)
        assert 0.0 <= p <= 1.0

    def test_erlang_c_saturated(self):
        from src.queue_model import erlang_c
        p = erlang_c(c=2, rho=1.0)
        assert p == 1.0

    def test_expected_wait_positive(self):
        from src.queue_model import expected_wait_erlang
        w = expected_wait_erlang(c=4, mu=3.5, rho=0.6)
        assert w >= 0

    def test_queue_state_utilisation(self):
        from src.queue_model import QueueState
        s = QueueState(station_id=1, n_chargers=4, arrival_rate=8.0, service_rate=3.5)
        assert abs(s.utilisation - 8.0 / (4 * 3.5)) < 1e-6

    def test_queue_estimate_keys(self):
        from src.queue_model import EVQueueModel, QueueState
        m = EVQueueModel()
        s = QueueState(station_id=1, n_chargers=6, arrival_rate=8.0, service_rate=3.5,
                       ml_wait_pred=12.5)
        out = m.estimate_wait(s)
        for key in ["blended_wait_min","erlang_wait_min","ml_wait_min",
                    "utilisation","availability","prob_wait"]:
            assert key in out, f"Missing key: {key}"

    def test_blended_wait_clipped(self):
        from src.queue_model import EVQueueModel, QueueState
        m = EVQueueModel()
        s = QueueState(station_id=1, n_chargers=1, arrival_rate=50.0,
                       service_rate=3.5, ml_wait_pred=200.0)
        out = m.estimate_wait(s)
        assert 0 <= out["blended_wait_min"] <= 120

    def test_arrival_rate_peak(self):
        from src.queue_model import EVQueueModel
        m   = EVQueueModel()
        off = m.infer_arrival_rate(hour=3,  is_weekend=False, traffic_index=0.2)
        pk  = m.infer_arrival_rate(hour=18, is_weekend=False, traffic_index=0.8)
        assert pk > off

    def test_availability_labels(self):
        from src.queue_model import EVQueueModel
        m = EVQueueModel()
        assert m._availability_label(0.2) == "Available"
        assert m._availability_label(0.5) == "Moderate"
        assert m._availability_label(0.8) == "Busy"
        assert m._availability_label(0.95)== "Full"


# ─── ML Model ─────────────────────────────────────────────────────────────────

class TestMLModel:

    @pytest.fixture(scope="class")
    def trained_rf(self, sample_sessions, sample_stations):
        from src.feature_engineering import (add_temporal_features, add_traffic_proxy,
                                              add_weather_proxy, add_station_features,
                                              add_historical_features, add_rolling_features,
                                              add_interaction_features)
        from src.data_preprocessing import merge_station_meta, encode_categoricals
        from src.models.ml_model import WaitTimeRFModel
        from config.config import FEATURE_COLS

        df = merge_station_meta(sample_sessions.copy(), sample_stations)
        for col in ["traffic_index","temperature","precipitation","nearby_pois",
                    "historical_avg_wait","rolling_mean_1h","rolling_mean_3h",
                    "rolling_std_3h","station_capacity"]:
            if col not in df.columns:
                df[col] = 0.0
        df = encode_categoricals(df)
        df = add_temporal_features(df)
        df = add_traffic_proxy(df)
        df = add_weather_proxy(df)
        df = add_station_features(df)
        df = add_historical_features(df)
        df = add_rolling_features(df)
        df = add_interaction_features(df)

        feat = [c for c in FEATURE_COLS if c in df.columns]
        X, y = df[feat], df["wait_time_minutes"]

        model = WaitTimeRFModel().build()
        model.fit(X, y)
        return model, feat, df

    def test_rf_predicts_positives(self, trained_rf):
        model, feat, df = trained_rf
        preds = model.predict(df[feat].head(10))
        assert (preds >= 0).all()

    def test_rf_output_length(self, trained_rf):
        model, feat, df = trained_rf
        n     = 25
        preds = model.predict(df[feat].head(n))
        assert len(preds) == n

    def test_rf_evaluate_metrics(self, trained_rf):
        model, feat, df = trained_rf
        y_pred  = model.predict(df[feat])
        metrics = WaitTimeRFModel.evaluate_metrics(df["wait_time_minutes"].values, y_pred)
        assert "MAE" in metrics and "R2" in metrics
        assert metrics["MAE"] >= 0

    def test_rf_feature_importance(self, trained_rf):
        model, feat, df = trained_rf
        fi = model.feature_importance()
        assert len(fi) == len(feat)
        assert fi["importance"].sum() == pytest.approx(1.0, abs=0.01)


# ─── Recommendation ───────────────────────────────────────────────────────────

class TestRecommender:

    def test_ranks_by_score(self):
        from src.recommendation import StationRecommender, UserPreferences
        df = pd.DataFrame([
            {"station_id":1,"name":"A","lat":37.775,"lon":-122.415,
             "blended_wait_min":3,"n_chargers":6,"availability":"Available",
             "connector_types":"CCS","is_free":True},
            {"station_id":2,"name":"B","lat":37.770,"lon":-122.420,
             "blended_wait_min":25,"n_chargers":4,"availability":"Busy",
             "connector_types":"J1772","is_free":False},
        ])
        rec     = StationRecommender(UserPreferences())
        results = rec.recommend(df, user_lat=37.775, user_lon=-122.415, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].rank  == 1

    def test_max_distance_filter(self):
        from src.recommendation import StationRecommender, UserPreferences
        df = pd.DataFrame([
            {"station_id":1,"name":"Near","lat":37.775,"lon":-122.415,
             "blended_wait_min":5,"n_chargers":4,"availability":"Available",
             "connector_types":"CCS","is_free":False},
            {"station_id":2,"name":"Far","lat":38.0,"lon":-122.0,   # ~30 km away
             "blended_wait_min":2,"n_chargers":8,"availability":"Available",
             "connector_types":"CCS","is_free":True},
        ])
        prefs   = UserPreferences(max_distance_km=5.0)
        rec     = StationRecommender(prefs)
        results = rec.recommend(df, user_lat=37.775, user_lon=-122.415)
        # Only Near should be returned
        assert all(r.station_id != 2 for r in results)

    def test_free_preference(self):
        from src.recommendation import StationRecommender, UserPreferences
        df = pd.DataFrame([
            {"station_id":1,"name":"Free","lat":37.775,"lon":-122.415,
             "blended_wait_min":15,"n_chargers":4,"availability":"Moderate",
             "connector_types":"CCS","is_free":True},
            {"station_id":2,"name":"Paid","lat":37.776,"lon":-122.416,
             "blended_wait_min":5,"n_chargers":8,"availability":"Available",
             "connector_types":"CCS","is_free":False},
        ])
        prefs   = UserPreferences(weight_free=0.5, weight_wait=0.3,
                                   weight_distance=0.1, weight_connector=0.1)
        rec     = StationRecommender(prefs)
        results = rec.recommend(df, user_lat=37.775, user_lon=-122.415)
        assert results[0].station_id == 1   # Free station wins with high free weight

    def test_summary_string(self):
        from src.recommendation import StationRecommender, UserPreferences
        df = pd.DataFrame([{
            "station_id":1,"name":"Test","lat":37.775,"lon":-122.415,
            "blended_wait_min":5,"n_chargers":4,"availability":"Available",
            "connector_types":"CCS","is_free":False,
        }])
        rec     = StationRecommender(UserPreferences())
        results = rec.recommend(df, user_lat=37.775, user_lon=-122.415)
        summary = rec.summary(results)
        assert "Test" in summary
        assert "#1"   in summary


# ─── Alert Service ────────────────────────────────────────────────────────────

class TestAlerts:

    def test_alert_triggers_above_threshold(self):
        from alerts.notification_service import NotificationService, AlertEvent
        svc = NotificationService(threshold_min=10)
        evt = AlertEvent(station_id=1, station_name="Test", wait_min=15.0,
                         availability="Busy")
        triggered = svc.check_and_notify(evt)
        assert triggered is True

    def test_alert_no_trigger_below_threshold(self):
        from alerts.notification_service import NotificationService, AlertEvent
        svc = NotificationService(threshold_min=20)
        evt = AlertEvent(station_id=1, station_name="Test", wait_min=8.0,
                         availability="Available")
        triggered = svc.check_and_notify(evt)
        assert triggered is False

    def test_alert_history(self):
        from alerts.notification_service import NotificationService, AlertEvent
        svc = NotificationService(threshold_min=5)
        for i in range(3):
            svc.check_and_notify(AlertEvent(station_id=i+1, station_name=f"S{i+1}",
                                             wait_min=20.0, availability="Busy"))
        assert len(svc.alert_history()) == 3


# ─── Config ────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_all_paths_exist(self):
        from config.config import RAW_DIR, PROC_DIR, CACHE_DIR, MODEL_DIR
        for d in [RAW_DIR, PROC_DIR, CACHE_DIR, MODEL_DIR]:
            assert d.exists(), f"Directory missing: {d}"

    def test_rf_params_valid(self):
        from config.config import RF_PARAMS
        assert RF_PARAMS["n_estimators"] > 0
        assert RF_PARAMS["max_depth"]    > 0
        assert "random_state" in RF_PARAMS

    def test_feature_cols_non_empty(self):
        from config.config import FEATURE_COLS, TARGET_COL
        assert len(FEATURE_COLS) >= 10
        assert isinstance(TARGET_COL, str)

    def test_queue_params(self):
        from config.config import QUEUE_SERVICE_RATE, QUEUE_ARRIVAL_WEIGHT
        assert QUEUE_SERVICE_RATE  > 0
        assert 0 < QUEUE_ARRIVAL_WEIGHT < 1
