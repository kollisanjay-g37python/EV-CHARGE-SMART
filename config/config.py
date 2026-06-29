"""
config/config.py
Central configuration for EV Charging Smart System.
All API keys, paths, model hyperparameters, and constants live here.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "real_time_cache"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ─── API Keys (set via environment variables) ────────────────────────────────
# Open Charge Map: https://openchargemap.org/site/develop/api
OPEN_CHARGE_MAP_API_KEY = os.getenv("OPEN_CHARGE_MAP_API_KEY")

# TomTom Traffic: https://developer.tomtom.com/traffic-api
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# Google Maps: https://developers.google.com/maps/documentation/directions/get-api-key
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# OpenWeatherMap: https://openweathermap.org/api
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ─── API Endpoints ────────────────────────────────────────────────────────────
OCM_BASE_URL = "https://api.openchargemap.io/v3"
TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4"
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# ─── Data Collection Settings ────────────────────────────────────────────────
DEFAULT_LOCATION = {"lat": 37.7749, "lng": -122.4194}   # San Francisco
DEFAULT_RADIUS_KM = 50
MAX_STATIONS_PER_CALL = 500
CACHE_TTL_SECONDS = 300        # 5 minutes for real-time data

# ─── Kaggle Dataset Paths (after manual download) ───────────────────────────
# https://www.kaggle.com/datasets/risheepanchal/global-ev-charging-stations-dataset
KAGGLE_STATIONS_CSV = RAW_DIR / "global_ev_charging_stations.csv"

# https://www.kaggle.com/datasets/datasetengineer/ev-charging-load-dataset-and-optimal-routing
KAGGLE_LOAD_CSV = RAW_DIR / "ev_charging_load_dataset.csv"

# https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
KAGGLE_ENERGY_CSV = RAW_DIR / "hourly_energy_consumption.csv"

# https://www.kaggle.com/datasets/salader/ev-demand-prediction
KAGGLE_DEMAND_CSV = RAW_DIR / "ev_demand_prediction.csv"

# ─── Processed Data Paths ────────────────────────────────────────────────────
PROCESSED_STATIONS_CSV = PROCESSED_DIR / "stations_processed.csv"
PROCESSED_SESSIONS_CSV = PROCESSED_DIR / "sessions_processed.csv"
PROCESSED_FEATURES_CSV = PROCESSED_DIR / "features_engineered.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ─── Model Paths ─────────────────────────────────────────────────────────────
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ─── ML Hyperparameters ──────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "sequence_length": 24,      # 24-hour look-back window
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 32,
    "dropout_rate": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "patience": 10,             # EarlyStopping patience
}

TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# ─── Feature Engineering Settings ───────────────────────────────────────────
TARGET_COLUMN = "wait_time_minutes"
FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "is_weekend", "month",
    "traffic_score", "station_utilization", "queue_size",
    "available_ports", "total_ports", "avg_session_duration_min",
    "temperature_c", "precipitation_mm", "connector_type_encoded",
    "distance_to_city_center_km", "nearby_stations_count",
    "rolling_mean_1h", "rolling_mean_3h", "rolling_std_1h",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

CYCLIC_FEATURES = {
    "hour_of_day": 24,
    "day_of_week": 7,
    "month": 12,
}

# ─── Queue Model (M/M/c) ─────────────────────────────────────────────────────
DEFAULT_SERVICE_RATE_PER_HOUR = 3.0   # sessions per port per hour
DEFAULT_ARRIVAL_RATE_PER_HOUR = 8.0   # vehicles per hour

# ─── FastAPI Backend ─────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]

# ─── Streamlit ───────────────────────────────────────────────────────────────
STREAMLIT_PORT = 8501
MAP_DEFAULT_ZOOM = 12

# ─── Alerts & Notifications ──────────────────────────────────────────────────
ALERT_WAIT_THRESHOLD_MIN = 30      # Alert if predicted wait > 30 min
ALERT_UTILIZATION_THRESHOLD = 0.9  # Alert if station > 90% utilized
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
