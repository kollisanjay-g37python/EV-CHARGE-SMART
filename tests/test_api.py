"""
tests/test_api.py
FastAPI integration tests using httpx TestClient.
Tests all /api/v1/* endpoints for correct status codes,
response schema, and prediction logic.

Run:
  pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from src.predict import PredictionEngine


# ─── Client Fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "rf_loaded" in data
        assert "lstm_loaded" in data

    def test_health_response_time(self, client):
        import time
        t0 = time.time()
        resp = client.get("/health")
        elapsed = time.time() - t0
        assert resp.status_code == 200
        assert elapsed < 2.0, "Health check too slow"


# ─── Stations ─────────────────────────────────────────────────────────────────

class TestStationsEndpoint:
    def test_list_stations_default(self, client):
        resp = client.get("/api/v1/stations")
        assert resp.status_code == 200
        data = resp.json()
        assert "stations" in data
        assert "total" in data
        assert len(data["stations"]) > 0

    def test_list_stations_limit(self, client):
        resp = client.get("/api/v1/stations?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()["stations"]) <= 5

    def test_list_stations_filter_status(self, client):
        resp = client.get("/api/v1/stations?status=Operational")
        assert resp.status_code == 200
        for s in resp.json()["stations"]:
            assert "Operational" in s["status"]

    def test_list_stations_with_location_sort(self, client):
        resp = client.get("/api/v1/stations?lat=37.7749&lng=-122.4194")
        assert resp.status_code == 200
        stations = resp.json()["stations"]
        if len(stations) > 1:
            # Should be distance-sorted
            assert "distance_km" in stations[0]
            dists = [s["distance_km"] for s in stations if "distance_km" in s]
            assert dists == sorted(dists)

    def test_get_single_station(self, client):
        resp = client.get("/api/v1/stations/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["station_id"] == 1
        assert "prediction" in data

    def test_get_missing_station(self, client):
        resp = client.get("/api/v1/stations/99999")
        assert resp.status_code == 404

    def test_list_stations_timestamp(self, client):
        resp = client.get("/api/v1/stations")
        assert "timestamp" in resp.json()


# ─── Prediction ───────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    VALID_PAYLOAD = {
        "station_id": 1,
        "num_ports": 8,
        "available_ports": 3,
        "queue_size": 4,
        "hour": 18,
        "day_of_week": 2,
        "traffic_score": 0.72,
        "temperature_c": 22.5,
        "precipitation_mm": 0.0,
        "connector_type": "CCS",
    }

    def test_predict_valid(self, client):
        resp = client.post("/api/v1/predict", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_wait_min" in data
        assert data["predicted_wait_min"] >= 0
        assert "recommendation" in data
        assert "timestamp" in data

    def test_predict_recommendation_labels(self, client):
        resp = client.post("/api/v1/predict", json=self.VALID_PAYLOAD)
        valid_labels = {"GO_NOW", "GOOD_TIME", "MODERATE_WAIT", "LONG_WAIT", "AVOID"}
        assert resp.json()["recommendation"] in valid_labels

    def test_predict_utilization_range(self, client):
        resp = client.post("/api/v1/predict", json=self.VALID_PAYLOAD)
        util = resp.json()["utilization_pct"]
        assert 0 <= util <= 150   # allow slight over-capacity

    def test_predict_missing_required_field(self, client):
        bad = self.VALID_PAYLOAD.copy()
        del bad["num_ports"]
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_available_ports_exceeds_total(self, client):
        bad = self.VALID_PAYLOAD.copy()
        bad["available_ports"] = 20   # more than num_ports=8
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_traffic_out_of_range(self, client):
        bad = self.VALID_PAYLOAD.copy()
        bad["traffic_score"] = 1.5   # > 1.0
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_hour_out_of_range(self, client):
        bad = self.VALID_PAYLOAD.copy()
        bad["hour"] = 25
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_with_location(self, client):
        payload = {**self.VALID_PAYLOAD, "lat": 37.7749, "lng": -122.4194}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_low_load_go_now(self, client):
        """Very low queue + high availability should give GO_NOW."""
        payload = {
            "station_id": 1, "num_ports": 8, "available_ports": 7,
            "queue_size": 0, "hour": 3, "day_of_week": 1,
            "traffic_score": 0.05,
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        # May be GO_NOW or GOOD_TIME
        assert resp.json()["predicted_wait_min"] < 20


class TestBatchPredict:
    def test_batch_multiple_stations(self, client):
        payload = [
            {"station_id": 1, "num_ports": 8, "available_ports": 3,
             "queue_size": 2, "hour": 8, "day_of_week": 0, "traffic_score": 0.6},
            {"station_id": 2, "num_ports": 4, "available_ports": 1,
             "queue_size": 5, "hour": 18, "day_of_week": 4, "traffic_score": 0.85},
        ]
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list(self, client):
        resp = client.post("/api/v1/predict/batch", json=[])
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ─── Recommendation ───────────────────────────────────────────────────────────

class TestRecommendEndpoint:
    def test_recommend_basic(self, client):
        payload = {
            "user_lat": 37.7749,
            "user_lng": -122.4194,
            "priority": "balanced",
            "top_n": 5,
        }
        resp = client.post("/api/v1/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 5

    def test_recommend_sorted_by_score(self, client):
        payload = {"user_lat": 37.7749, "user_lng": -122.4194, "top_n": 5}
        resp = client.post("/api/v1/recommend", json=payload)
        recs = resp.json()["recommendations"]
        if len(recs) > 1:
            scores = [r["composite_score"] for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_speed_priority(self, client):
        payload = {
            "user_lat": 37.7749, "user_lng": -122.4194,
            "priority": "speed", "top_n": 3,
        }
        resp = client.post("/api/v1/recommend", json=payload)
        assert resp.status_code == 200

    def test_recommend_with_connector_filter(self, client):
        payload = {
            "user_lat": 37.7749, "user_lng": -122.4194,
            "connector_type": "CCS", "top_n": 5,
        }
        resp = client.post("/api/v1/recommend", json=payload)
        assert resp.status_code == 200

    def test_recommend_future_hour(self, client):
        payload = {
            "user_lat": 37.7749, "user_lng": -122.4194,
            "target_hour": 20, "top_n": 3,
        }
        resp = client.post("/api/v1/recommend", json=payload)
        assert resp.status_code == 200

    def test_recommend_response_has_routing_url(self, client):
        payload = {"user_lat": 37.7749, "user_lng": -122.4194, "top_n": 1}
        resp = client.post("/api/v1/recommend", json=payload)
        recs = resp.json()["recommendations"]
        if recs:
            assert "routing_url" in recs[0]


# ─── Queue Analysis ───────────────────────────────────────────────────────────

class TestQueueEndpoint:
    def test_queue_analysis(self, client):
        resp = client.get("/api/v1/queue/1?arrival_rate=8.0&service_rate=3.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_wait_min" in data
        assert "rho" in data
        assert "system_stable" in data
        assert data["avg_wait_min"] >= 0
        assert 0 < data["rho"] < 10

    def test_queue_missing_station(self, client):
        resp = client.get("/api/v1/queue/99999")
        assert resp.status_code == 404

    def test_queue_stable_flag(self, client):
        # Undersaturated: should be stable
        resp = client.get("/api/v1/queue/1?arrival_rate=2.0&service_rate=5.0")
        assert resp.json()["system_stable"] is True


# ─── Forecast ─────────────────────────────────────────────────────────────────

class TestForecastEndpoint:
    def test_forecast_returns_12_steps(self, client):
        resp = client.get("/api/v1/forecast/1")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 12

    def test_forecast_has_required_fields(self, client):
        resp = client.get("/api/v1/forecast/1")
        for point in resp.json()["forecast"]:
            assert "hour" in point
            assert "predicted_wait_min" in point
            assert "recommendation" in point

    def test_forecast_non_negative_waits(self, client):
        resp = client.get("/api/v1/forecast/1")
        waits = [p["predicted_wait_min"] for p in resp.json()["forecast"]]
        assert all(w >= 0 for w in waits)


# ─── Metrics & Capacity ───────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_capacity_plan(self, client):
        resp = client.get("/api/v1/capacity-plan?arrival_rate=10.0&service_rate=3.0&target_wait_min=15.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_ports" in data
        assert data["recommended_ports"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
