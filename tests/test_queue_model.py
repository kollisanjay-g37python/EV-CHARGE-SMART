# tests/test_queue_model.py
"""
Unit tests for the Erlang-C queue model.
Run with: pytest tests/test_queue_model.py -v
"""

import pytest
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.queue_model import (
    EVQueueModel, QueueState, erlang_c, expected_wait_erlang
)


# ─── erlang_c ─────────────────────────────────────────────────────────────────

class TestErlangC:
    def test_stable_queue_returns_probability(self):
        p = erlang_c(c=4, rho=0.5)
        assert 0 <= p <= 1

    def test_saturated_returns_one(self):
        assert erlang_c(c=2, rho=1.0) == 1.0

    def test_oversaturated_returns_one(self):
        assert erlang_c(c=2, rho=1.5) == 1.0

    def test_low_utilisation_low_prob(self):
        p = erlang_c(c=8, rho=0.1)
        assert p < 0.1

    def test_high_utilisation_high_prob(self):
        p = erlang_c(c=2, rho=0.95)
        assert p > 0.8

    def test_single_server(self):
        p = erlang_c(c=1, rho=0.5)
        assert 0 <= p <= 1

    def test_many_servers(self):
        p = erlang_c(c=20, rho=0.3)
        assert p < 0.05


# ─── expected_wait_erlang ─────────────────────────────────────────────────────

class TestExpectedWaitErlang:
    def test_stable_returns_finite(self):
        w = expected_wait_erlang(c=4, mu=3.5, rho=0.6)
        assert 0 <= w < 60

    def test_saturated_returns_inf(self):
        w = expected_wait_erlang(c=2, mu=3.5, rho=1.0)
        assert math.isinf(w)

    def test_zero_utilisation_returns_zero(self):
        w = expected_wait_erlang(c=4, mu=3.5, rho=0.0)
        assert w == 0.0

    def test_more_servers_less_wait(self):
        w2 = expected_wait_erlang(c=2, mu=3.5, rho=0.7)
        w4 = expected_wait_erlang(c=4, mu=3.5, rho=0.7)
        assert w4 < w2


# ─── QueueState ───────────────────────────────────────────────────────────────

class TestQueueState:
    def test_utilisation_formula(self):
        state = QueueState(
            station_id=1, n_chargers=4,
            arrival_rate=8.0, service_rate=3.5,
        )
        expected = 8.0 / (4 * 3.5)
        assert abs(state.utilisation - expected) < 1e-6

    def test_stable_when_rho_lt_1(self):
        state = QueueState(station_id=1, n_chargers=6,
                           arrival_rate=6.0, service_rate=3.5)
        assert state.is_stable

    def test_unstable_when_rho_gte_1(self):
        state = QueueState(station_id=1, n_chargers=1,
                           arrival_rate=10.0, service_rate=3.5)
        assert not state.is_stable


# ─── EVQueueModel ─────────────────────────────────────────────────────────────

class TestEVQueueModel:
    @pytest.fixture
    def model(self):
        return EVQueueModel(service_rate=3.5)

    def test_estimate_wait_returns_dict(self, model):
        state = QueueState(station_id=1, n_chargers=6,
                           arrival_rate=8.0, service_rate=3.5,
                           ml_wait_pred=12.0)
        result = model.estimate_wait(state)
        assert isinstance(result, dict)

    def test_required_output_keys(self, model):
        state = QueueState(station_id=1, n_chargers=4,
                           arrival_rate=5.0, service_rate=3.5)
        result = model.estimate_wait(state)
        for key in ["blended_wait_min", "utilisation",
                    "availability", "prob_wait", "erlang_wait_min"]:
            assert key in result

    def test_blended_wait_clipped(self, model):
        state = QueueState(station_id=1, n_chargers=1,
                           arrival_rate=20.0, service_rate=3.5,
                           ml_wait_pred=200.0)
        result = model.estimate_wait(state)
        assert result["blended_wait_min"] <= 120

    def test_blended_wait_non_negative(self, model):
        state = QueueState(station_id=1, n_chargers=10,
                           arrival_rate=1.0, service_rate=3.5,
                           ml_wait_pred=0.0)
        result = model.estimate_wait(state)
        assert result["blended_wait_min"] >= 0

    def test_availability_labels(self, model):
        cases = [
            (0.2, "Available"),
            (0.5, "Moderate"),
            (0.8, "Busy"),
            (0.95, "Full"),
        ]
        for rho, expected_label in cases:
            label = model._availability_label(rho)
            assert label == expected_label

    def test_ml_pred_blends_correctly(self, model):
        state = QueueState(station_id=1, n_chargers=6,
                           arrival_rate=4.0, service_rate=3.5,
                           ml_wait_pred=20.0)
        result = model.estimate_wait(state)
        # ML weight = 0.6, so blended should be influenced by 20.0
        assert result["blended_wait_min"] > result["erlang_wait_min"] * 0.5

    def test_queue_length_increases_wait(self, model):
        state_no_q = QueueState(station_id=1, n_chargers=4,
                                arrival_rate=6.0, service_rate=3.5,
                                queue_length=0)
        state_q5   = QueueState(station_id=1, n_chargers=4,
                                arrival_rate=6.0, service_rate=3.5,
                                queue_length=5)
        r_no_q = model.estimate_wait(state_no_q)
        r_q5   = model.estimate_wait(state_q5)
        assert r_q5["blended_wait_min"] >= r_no_q["blended_wait_min"]

    @pytest.mark.parametrize("hour,is_weekend,expected_trend", [
        (8,  False, "high"),
        (18, False, "high"),
        (3,  False, "low"),
        (3,  True,  "low"),
        (13, False, "medium"),
    ])
    def test_infer_arrival_rate(self, model, hour, is_weekend, expected_trend):
        rate = model.infer_arrival_rate(hour, is_weekend, traffic_index=0.4)
        assert rate > 0
        if expected_trend == "high":
            assert rate > 5.0
        elif expected_trend == "low":
            assert rate < 4.0

    def test_batch_estimate_returns_dataframe(self, model, sample_stations):
        import pandas as pd
        result = model.batch_estimate(
            sample_stations, hour=9, is_weekend=False,
            traffic_index=0.6,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_stations)
        assert "blended_wait_min" in result.columns
