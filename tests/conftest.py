"""
tests/conftest.py
Shared pytest fixtures and configuration for all test modules.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from config.config import TARGET_COLUMN


@pytest.fixture(scope="session")
def sample_sessions_large():
    from src.data_collection import KaggleDatasetLoader
    loader = KaggleDatasetLoader()
    return loader._generate_synthetic_sessions(n=2000)


@pytest.fixture(scope="session")
def sample_stations_df():
    np.random.seed(0)
    n = 25
    return pd.DataFrame({
        "station_id": range(1, n + 1),
        "name": [f"Station {i}" for i in range(1, n + 1)],
        "lat": 37.77 + np.random.uniform(-0.05, 0.05, n),
        "lng": -122.42 + np.random.uniform(-0.08, 0.08, n),
        "num_ports": np.random.randint(2, 16, n),
        "available_ports": np.random.randint(0, 8, n),
        "queue_size": np.random.randint(0, 7, n),
        "connector_type": np.random.choice(["CCS", "CHAdeMO", "Type 2", "Tesla"], n),
        "power_kw": np.random.choice([7.2, 50.0, 150.0, 350.0], n).astype(float),
        "status": np.random.choice(["Operational", "Operational", "Partial", "Offline"], n),
        "operator": np.random.choice(["Tesla", "ChargePoint", "EVgo", "Blink"], n),
        "traffic_score": np.random.uniform(0.1, 0.9, n),
        "wait_time_minutes": np.random.uniform(0, 45, n),
    })


@pytest.fixture(scope="session")
def default_user_location():
    return {"lat": 37.7749, "lng": -122.4194}


@pytest.fixture(scope="session")
def trained_rf_model(sample_sessions_large):
    from src.feature_engineering import FeatureEngineer
    from src.models.ml_model import WaitTimeRFModel
    from config.config import FEATURE_COLUMNS

    fe = FeatureEngineer()
    df = fe.run(sample_sessions_large, save=False)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feat_cols:
        feat_cols = list(df.select_dtypes(include=[np.number]).columns)
        feat_cols = [c for c in feat_cols if c != TARGET_COLUMN]

    X = df[feat_cols]
    y = df[TARGET_COLUMN]
    rf = WaitTimeRFModel()
    rf.build().train(X, y, feature_cols=feat_cols)
    return rf


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "requires_tf: mark test as requiring TensorFlow")
