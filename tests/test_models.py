"""
tests/test_models.py
Unit tests for:
  - WaitTimeRFModel  (train, predict, save, load, uncertainty)
  - LSTMWaitTimeModel (build, predict, multistep)
  - MMcQueueModel  (Erlang-C, wait times, capacity planning)

Run:
  pytest tests/test_models.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import tempfile

from src.models.ml_model import WaitTimeRFModel, WaitTimeGBModel
from src.models.lstm_model import (
    LSTMWaitTimeModel, create_sequences, train_val_split_temporal
)
from src.queue_model import MMcQueueModel, QueueState
from config.config import LSTM_PARAMS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_tabular():
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "hour_of_day": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "traffic_score": np.random.uniform(0, 1, n),
        "station_utilization": np.random.uniform(0, 1, n),
        "queue_size": np.random.randint(0, 10, n),
        "is_weekend": np.random.randint(0, 2, n),
        "temperature_c": np.random.normal(18, 6, n),
        "hour_sin": np.sin(2 * np.pi * np.random.randint(0, 24, n) / 24),
        "hour_cos": np.cos(2 * np.pi * np.random.randint(0, 24, n) / 24),
    })
    y = pd.Series(
        X["queue_size"] * 8 + X["traffic_score"] * 15
        + np.random.normal(0, 3, n)
    ).clip(0)
    return X, y


@pytest.fixture(scope="module")
def dummy_sequences():
    np.random.seed(42)
    seq_len = LSTM_PARAMS["sequence_length"]
    n_feats = 8
    n = 300
    X = np.random.rand(n, seq_len, n_feats).astype(np.float32)
    y = np.random.uniform(0, 45, n).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def trained_rf(dummy_tabular):
    X, y = dummy_tabular
    rf = WaitTimeRFModel()
    rf.build().train(X, y, feature_cols=list(X.columns))
    return rf


# ─── Random Forest ────────────────────────────────────────────────────────────

class TestWaitTimeRFModel:
    def test_build(self):
        rf = WaitTimeRFModel()
        rf.build()
        assert rf.model is not None

    def test_train_output_shape(self, trained_rf, dummy_tabular):
        X, y = dummy_tabular
        preds = trained_rf.predict(X)
        assert len(preds) == len(X)

    def test_predict_non_negative(self, trained_rf, dummy_tabular):
        X, _ = dummy_tabular
        preds = trained_rf.predict(X)
        assert (preds >= 0).all(), "Predictions must be non-negative"

    def test_predict_reasonable_range(self, trained_rf, dummy_tabular):
        X, y = dummy_tabular
        preds = trained_rf.predict(X)
        assert preds.max() < 500, "Predictions suspiciously large"
        assert preds.mean() < 200

    def test_feature_importances(self, trained_rf):
        assert trained_rf.feature_importances_ is not None
        assert len(trained_rf.feature_importances_) > 0
        assert abs(trained_rf.feature_importances_.sum() - 1.0) < 0.001

    def test_predict_with_uncertainty(self, trained_rf, dummy_tabular):
        X, _ = dummy_tabular
        mean, lower, upper = trained_rf.predict_with_uncertainty(X.head(10))
        assert len(mean) == 10
        assert (upper >= mean).all()
        assert (mean >= lower).all()
        assert (lower >= 0).all()

    def test_save_and_load(self, trained_rf, dummy_tabular):
        X, _ = dummy_tabular
        preds_before = trained_rf.predict(X.head(5))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            trained_rf.save(tmp_path)
            loaded = WaitTimeRFModel()
            loaded.load(tmp_path)
            preds_after = loaded.predict(X.head(5))
            np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=4)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_feature_columns_handled(self, trained_rf):
        """Model should handle missing feature columns gracefully (fills with 0)."""
        X_partial = pd.DataFrame({
            "hour_of_day": [8, 18],
            "traffic_score": [0.3, 0.8],
        })
        preds = trained_rf.predict(X_partial)
        assert len(preds) == 2
        assert (preds >= 0).all()

    def test_r2_above_threshold(self, trained_rf, dummy_tabular):
        from sklearn.metrics import r2_score
        X, y = dummy_tabular
        preds = trained_rf.predict(X)
        r2 = r2_score(y, preds)
        assert r2 > 0.5, f"R² too low: {r2:.4f}"

    def test_cross_validate(self, dummy_tabular):
        X, y = dummy_tabular
        rf = WaitTimeRFModel()
        scores = rf.cross_validate(X, y, feature_cols=list(X.columns), cv=3)
        assert "root_mean_squared_error" in scores
        assert "r2" in scores
        assert scores["r2"] > 0.3


class TestWaitTimeGBModel:
    def test_gbm_train_predict(self, dummy_tabular):
        X, y = dummy_tabular
        gbm = WaitTimeGBModel()
        gbm.train(X, y, feature_cols=list(X.columns))
        preds = gbm.predict(X.head(10))
        assert len(preds) == 10
        assert (preds >= 0).all()


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class TestLSTMWaitTimeModel:
    def test_mock_predict(self, dummy_sequences):
        X, _ = dummy_sequences
        lstm = LSTMWaitTimeModel()
        # No TF — falls back to mock
        preds = lstm.predict(X[:5])
        assert len(preds) == 5

    def test_mock_predict_non_negative(self, dummy_sequences):
        X, _ = dummy_sequences
        lstm = LSTMWaitTimeModel()
        preds = lstm.predict(X)
        assert (preds >= 0).all()

    def test_mock_multistep(self, dummy_sequences):
        X, _ = dummy_sequences
        lstm = LSTMWaitTimeModel()
        multi = lstm.predict_multistep(X[:1], steps=6)
        assert len(multi) == 6

    def test_create_sequences_shape(self, dummy_tabular):
        X, y = dummy_tabular
        df = X.copy()
        df["wait_time_minutes"] = y.values
        seq_len = 12
        X_seq, y_seq = create_sequences(df, list(X.columns), "wait_time_minutes", seq_len)
        assert X_seq.ndim == 3
        assert X_seq.shape[1] == seq_len
        assert X_seq.shape[2] == len(X.columns)
        assert len(X_seq) == len(y_seq)
        assert len(X_seq) == len(df) - seq_len

    def test_create_sequences_y_values(self, dummy_tabular):
        X, y = dummy_tabular
        df = X.copy()
        df["wait_time_minutes"] = y.values
        seq_len = 10
        X_seq, y_seq = create_sequences(df, list(X.columns), "wait_time_minutes", seq_len)
        # y_seq[0] should correspond to df["wait_time_minutes"][seq_len]
        assert y_seq[0] == pytest.approx(df["wait_time_minutes"].iloc[seq_len], abs=0.001)

    def test_train_val_split_temporal_order(self):
        X = np.arange(100).reshape(100, 1, 1).astype(float)
        y = np.arange(100).astype(float)
        X_tr, y_tr, X_v, y_v = train_val_split_temporal(X, y, val_ratio=0.2)
        assert len(X_tr) == 80
        assert len(X_v) == 20
        # Temporal order preserved (train before val)
        assert y_tr[-1] < y_v[0]

    def test_train_val_split_sizes(self, dummy_sequences):
        X, y = dummy_sequences
        X_tr, y_tr, X_v, y_v = train_val_split_temporal(X, y, val_ratio=0.15)
        total = len(X_tr) + len(X_v)
        assert total == len(X)
        val_ratio = len(X_v) / total
        assert 0.12 <= val_ratio <= 0.18


# ─── Queue Model ──────────────────────────────────────────────────────────────

class TestMMcQueueModel:
    def test_basic_wait_stable(self):
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=4, arrival_rate=8.0, service_rate=3.5)
        assert state.system_stable
        assert state.avg_wait_min >= 0
        assert 0 < state.rho < 1

    def test_unstable_system(self):
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=2, arrival_rate=20.0, service_rate=3.0)
        assert not state.system_stable

    def test_zero_queue_zero_wait(self):
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=8, arrival_rate=2.0, service_rate=4.0)
        assert state.system_stable
        assert state.avg_wait_min < 5.0   # very low load → minimal wait

    def test_rho_formula(self):
        """ρ = λ / (c·μ)"""
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=4, arrival_rate=8.0, service_rate=2.0)
        expected_rho = 8.0 / (4 * 2.0)
        assert state.rho == pytest.approx(expected_rho, abs=0.001)

    def test_erlang_c_bounds(self):
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=4, arrival_rate=8.0, service_rate=3.5)
        assert 0.0 <= state.erlang_c <= 1.0

    def test_avg_queue_little_law(self):
        """Lq ≈ λ · Wq (Little's Law)."""
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=4, arrival_rate=6.0, service_rate=3.0)
        if state.system_stable:
            wq_hr = state.avg_wait_min / 60
            lq_expected = state.arrival_rate * wq_hr
            assert abs(state.avg_queue_length - lq_expected) < 0.5

    def test_capacity_recommendation(self):
        qm = MMcQueueModel()
        rec = qm.recommend_capacity(arrival_rate=10.0, service_rate=3.0, target_wait_min=15.0)
        assert "recommended_ports" in rec
        assert rec["recommended_ports"] >= 1

    def test_capacity_recommendation_achieves_target(self):
        qm = MMcQueueModel()
        rec = qm.recommend_capacity(arrival_rate=5.0, service_rate=3.0, target_wait_min=20.0)
        assert rec["predicted_wait_min"] <= 20.0

    def test_sensitivity_analysis_stable(self):
        qm = MMcQueueModel()
        df = qm.sensitivity_analysis(num_ports=4, service_rate=3.0)
        assert len(df) > 0
        stable = df[df["system_stable"]]
        if len(stable) > 0:
            assert (stable["avg_wait_min"].dropna() >= 0).all()

    def test_sensitivity_monotonic_wait(self):
        """Wait time should increase as arrival rate increases."""
        qm = MMcQueueModel()
        lams = [2.0, 4.0, 6.0, 8.0, 10.0]
        df = qm.sensitivity_analysis(num_ports=4, service_rate=3.0, arrival_rates=lams)
        stable = df[df["system_stable"]].dropna(subset=["avg_wait_min"])
        if len(stable) >= 3:
            waits = stable["avg_wait_min"].values
            # Generally increasing (allow small non-monotonicity due to blending)
            assert waits[-1] > waits[0]

    def test_network_analysis(self, tmp_path):
        qm = MMcQueueModel()
        stations = pd.DataFrame({
            "station_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "num_ports": [4, 6, 2],
        })
        result = qm.analyze_network(stations)
        assert len(result) == 3
        assert "avg_wait_min" in result.columns
        assert "rho" in result.columns

    def test_to_dict_keys(self):
        qm = MMcQueueModel()
        state = qm.compute_wait(num_ports=4, arrival_rate=6.0, service_rate=3.0)
        d = state.to_dict()
        required_keys = [
            "station_id", "num_ports", "arrival_rate", "service_rate",
            "rho", "erlang_c", "avg_wait_min", "system_stable", "utilization_pct"
        ]
        for k in required_keys:
            assert k in d, f"Missing key in QueueState.to_dict(): {k}"

    def test_more_ports_lower_wait(self):
        """Adding ports should reduce wait time."""
        qm = MMcQueueModel()
        s2 = qm.compute_wait(num_ports=2, arrival_rate=4.0, service_rate=3.0)
        s6 = qm.compute_wait(num_ports=6, arrival_rate=4.0, service_rate=3.0)
        if s2.system_stable and s6.system_stable:
            assert s6.avg_wait_min < s2.avg_wait_min


# ─── Evaluate ─────────────────────────────────────────────────────────────────

class TestModelEvaluator:
    def test_compute_metrics(self):
        from src.evaluate import ModelEvaluator
        ev = ModelEvaluator()
        y_true = np.array([10, 20, 30, 25, 15])
        y_pred = np.array([11, 19, 28, 26, 14])
        metrics = ev.compute_metrics(y_true, y_pred, "Test")
        assert metrics["rmse_min"] >= 0
        assert metrics["mae_min"] >= 0
        assert -1.0 <= metrics["r2"] <= 1.0
        assert 0 <= metrics["within_5min_pct"] <= 100

    def test_compare_models(self):
        from src.evaluate import ModelEvaluator
        ev = ModelEvaluator()
        m1 = {"rmse_min": 4.2, "mae_min": 3.1, "r2": 0.887,
               "mape_pct": 15.2, "within_5min_pct": 71.3, "within_10min_pct": 88.0}
        m2 = {"rmse_min": 3.6, "mae_min": 2.8, "r2": 0.912,
               "mape_pct": 13.1, "within_5min_pct": 74.8, "within_10min_pct": 90.2}
        table = ev.compare_models({"RF": m1, "LSTM": m2})
        assert len(table) == 2
        assert "RMSE (min)" in table.columns

    def test_residual_analysis(self):
        from src.evaluate import ModelEvaluator
        ev = ModelEvaluator()
        y_true = np.random.uniform(5, 40, 100)
        y_pred = y_true + np.random.normal(0, 3, 100)
        df = ev.residual_analysis(y_true, y_pred)
        assert len(df) == 100
        assert "residual" in df.columns
        assert "abs_residual" in df.columns
        assert (df["abs_residual"] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
