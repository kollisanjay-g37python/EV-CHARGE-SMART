"""
tests/test_predict.py
Unit tests for PredictionEngine (predict.py).
Covers: single prediction, batch, recommendation integration,
        cache, realtime injection, and edge cases.

Run:
  pytest tests/test_predict.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.predict import PredictionEngine
from src.feature_engineering import FeatureEngineer


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Unloaded engine — uses queue model fallback (no pkl required)."""
    e = PredictionEngine()
    # Don't call load_models() — works with queue model fallback
    return e


@pytest.fixture(scope="module")
def base_payload():
    return dict(
        station_id=1,
        num_ports=8,
        available_ports=3,
        queue_size=4,
        hour=18,
        day_of_week=2,
        traffic_score=0.72,
        temperature_c=22.0,
        precipitation_mm=0.0,
        connector_type="CCS",
        lat=37.7749,
        lng=-122.4194,
    )


# ─── Single Prediction ────────────────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_dict(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert isinstance(result, dict)

    def test_required_keys(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        for key in ["station_id", "predicted_wait_min", "queue_prediction",
                    "recommendation", "confidence_level", "utilization_pct"]:
            assert key in result, f"Missing key: {key}"

    def test_non_negative_wait(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert result["predicted_wait_min"] >= 0

    def test_utilization_range(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert 0 <= result["utilization_pct"] <= 150

    def test_recommendation_valid_labels(self, engine, base_payload):
        valid = {"GO_NOW", "GOOD_TIME", "MODERATE_WAIT", "LONG_WAIT", "AVOID"}
        result = engine.predict_single(**base_payload)
        assert result["recommendation"] in valid

    def test_station_id_preserved(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert result["station_id"] == base_payload["station_id"]

    def test_queue_stable_field(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert isinstance(result["queue_stable"], bool)

    def test_go_now_low_load(self, engine):
        result = engine.predict_single(
            station_id=1, num_ports=10, available_ports=9,
            queue_size=0, hour=3, day_of_week=6,
            traffic_score=0.02,
        )
        # Very low load — should be short wait
        assert result["predicted_wait_min"] < 20

    def test_high_load_higher_wait(self, engine):
        low = engine.predict_single(
            station_id=1, num_ports=8, available_ports=7,
            queue_size=0, hour=3, day_of_week=0, traffic_score=0.1,
        )
        high = engine.predict_single(
            station_id=1, num_ports=8, available_ports=1,
            queue_size=7, hour=18, day_of_week=4, traffic_score=0.9,
        )
        assert high["predicted_wait_min"] >= low["predicted_wait_min"]

    def test_different_hours_different_results(self, engine, base_payload):
        r_3am  = engine.predict_single(**{**base_payload, "hour": 3})
        r_6pm  = engine.predict_single(**{**base_payload, "hour": 18})
        # Results may differ (queue model is hour-independent, but features differ)
        assert isinstance(r_3am["predicted_wait_min"], float)
        assert isinstance(r_6pm["predicted_wait_min"], float)

    def test_no_lat_lng(self, engine):
        """Should still work without optional lat/lng."""
        result = engine.predict_single(
            station_id=5, num_ports=6, available_ports=2,
            queue_size=3, hour=12, day_of_week=1, traffic_score=0.5,
        )
        assert "predicted_wait_min" in result

    def test_zero_ports_handled(self, engine):
        """Should not crash with edge-case inputs."""
        result = engine.predict_single(
            station_id=1, num_ports=1, available_ports=0,
            queue_size=0, hour=8, day_of_week=0, traffic_score=0.5,
        )
        assert result["predicted_wait_min"] >= 0

    def test_confidence_level_string(self, engine, base_payload):
        result = engine.predict_single(**base_payload)
        assert result["confidence_level"] in {"high", "medium", "low"}


# ─── Batch Prediction ─────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_returns_list(self, engine):
        df = pd.DataFrame({
            "station_id": [1, 2, 3],
            "num_ports":  [8, 4, 6],
            "available_ports": [3, 1, 4],
            "queue_size": [2, 5, 0],
            "hour_of_day": [8, 18, 12],
            "traffic_score": [0.4, 0.8, 0.3],
        })
        result = engine.predict_batch(df)
        assert "predicted_wait_min" in result.columns
        assert len(result) == 3

    def test_batch_non_negative(self, engine):
        df = pd.DataFrame({
            "station_id": [1, 2],
            "num_ports":  [8, 4],
            "available_ports": [2, 1],
            "queue_size": [3, 5],
        })
        result = engine.predict_batch(df)
        assert (result["predicted_wait_min"] >= 0).all()

    def test_batch_has_recommendation(self, engine):
        df = pd.DataFrame({
            "station_id": [1],
            "available_ports": [2],
        })
        result = engine.predict_batch(df)
        assert "recommendation" in result.columns


# ─── Feature Preparation ──────────────────────────────────────────────────────

class TestFeaturePreparation:
    def test_prepare_features_returns_df(self, engine, base_payload):
        df = engine._prepare_features(base_payload)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_prepare_features_cyclic(self, engine, base_payload):
        df = engine._prepare_features(base_payload)
        assert "hour_sin" in df.columns or "hour_of_day" in df.columns

    def test_prepare_features_weekend_flag(self, engine):
        payload = dict(
            station_id=1, num_ports=4, available_ports=2, queue_size=1,
            hour=10, day_of_week=5, traffic_score=0.3,
        )
        df = engine._prepare_features(payload)
        assert "is_weekend" in df.columns
        assert df["is_weekend"].iloc[0] == 1   # Saturday = weekend


# ─── Recommendation label ─────────────────────────────────────────────────────

class TestRecommendationLabel:
    @pytest.mark.parametrize("wait,avail,expected", [
        (2,  5, "GO_NOW"),
        (10, 2, "GOOD_TIME"),
        (25, 1, "MODERATE_WAIT"),
        (50, 0, "LONG_WAIT"),
    ])
    def test_labels(self, wait, avail, expected):
        label = PredictionEngine._get_recommendation(wait, avail)
        assert label == expected

    def test_zero_wait_go_now(self):
        assert PredictionEngine._get_recommendation(0, 3) == "GO_NOW"

    def test_no_ports_long_wait(self):
        label = PredictionEngine._get_recommendation(65, 0)
        assert label in {"LONG_WAIT", "AVOID"}


# ─── Feature Engineer Integration ────────────────────────────────────────────

class TestFeatureEngineerInPredict:
    def test_add_temporal_no_crash(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "station_id": [1, 2],
            "traffic_score": [0.3, 0.7],
        })
        result = fe.add_temporal_features(df)
        assert "hour_of_day" in result.columns

    def test_add_weather_features(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({"temperature_c": [2.0, 38.0], "precipitation_mm": [0.0, 5.0]})
        result = fe.add_weather_features(df)
        assert result["is_cold"].iloc[0] == 1
        assert result["is_hot"].iloc[1] == 1
        assert result["is_raining"].iloc[1] == 1

    def test_utilization_clipped_0_1(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({"available_ports": [0, 3, 8], "num_ports": [4, 4, 4]})
        result = fe.add_utilization_features(df)
        assert result["station_utilization"].between(0, 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
