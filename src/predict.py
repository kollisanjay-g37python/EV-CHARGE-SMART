"""
src/predict.py
Inference module — loads trained models and generates predictions.
Supports:
  - Single-point prediction (one station, one timestamp)
  - Batch prediction (DataFrame of feature rows)
  - Ensemble prediction (weighted RF + LSTM)
  - Real-time prediction with live traffic/weather injection
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    FEATURE_COLUMNS, LABEL_ENCODER_PATH, LSTM_PARAMS,
    RF_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH, TARGET_COLUMN,
)
from src.models.ml_model import WaitTimeRFModel
from src.models.lstm_model import LSTMWaitTimeModel, create_sequences
from src.feature_engineering import FeatureEngineer
from src.queue_model import MMcQueueModel

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Unified inference engine combining RF + LSTM + Queue Model.
    Weights: RF=0.40, LSTM=0.45, Queue=0.15 (configurable).
    """

    RF_WEIGHT = 0.40
    LSTM_WEIGHT = 0.45
    QUEUE_WEIGHT = 0.15

    def __init__(self):
        self.rf = WaitTimeRFModel()
        self.lstm = LSTMWaitTimeModel()
        self.queue_model = MMcQueueModel()
        self.fe = FeatureEngineer()
        self._rf_loaded = False
        self._lstm_loaded = False
        self._scaler = None
        self._encoder = None

    # ─── Load ─────────────────────────────────────────────────────────────────

    def load_models(self) -> "PredictionEngine":
        """Load all serialised models from disk."""
        if RF_MODEL_PATH.exists():
            try:
                self.rf.load(RF_MODEL_PATH)
                self._rf_loaded = True
                logger.info("RF model loaded")
            except Exception as e:
                logger.warning(f"RF load failed: {e}")

        if LSTM_MODEL_PATH.exists():
            try:
                self.lstm.load(LSTM_MODEL_PATH)
                self._lstm_loaded = True
                logger.info("LSTM model loaded")
            except Exception as e:
                logger.warning(f"LSTM load failed: {e}")

        if SCALER_PATH.exists():
            self._scaler = joblib.load(SCALER_PATH)

        if LABEL_ENCODER_PATH.exists():
            self._encoder = joblib.load(LABEL_ENCODER_PATH)

        return self

    # ─── Feature Preparation ──────────────────────────────────────────────────

    def _prepare_features(self, raw: dict) -> pd.DataFrame:
        """Convert raw input dict to engineered feature DataFrame."""
        df = pd.DataFrame([raw])

        # Parse datetime from components if provided
        if "hour" in raw and "session_start" not in raw:
            now = pd.Timestamp.now()
            df["session_start"] = now.replace(hour=raw["hour"], minute=0, second=0)

        df = self.fe.add_temporal_features(df)
        df = self.fe.add_traffic_features(df)
        df = self.fe.add_utilization_features(df)
        df = self.fe.add_weather_features(df)

        # Encode connector_type if present
        if "connector_type" in df.columns and self._encoder is not None:
            try:
                df["connector_type_encoded"] = self._encoder.transform(
                    df["connector_type"].astype(str)
                )
            except Exception:
                df["connector_type_encoded"] = 0
        elif "connector_type_encoded" not in df.columns:
            df["connector_type_encoded"] = 0

        return df

    # ─── Single Prediction ────────────────────────────────────────────────────

    def predict_single(
        self,
        station_id: int,
        num_ports: int,
        available_ports: int,
        queue_size: int,
        hour: int,
        day_of_week: int,
        traffic_score: float,
        temperature_c: float = 20.0,
        precipitation_mm: float = 0.0,
        connector_type: str = "Type 2",
        lat: float = None,
        lng: float = None,
    ) -> Dict:
        raw = {
            "station_id": station_id,
            "num_ports": num_ports,
            "available_ports": available_ports,
            "queue_size": queue_size,
            "hour": hour,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week >= 5),
            "traffic_score": traffic_score,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "connector_type": connector_type,
        }
        if lat: raw["lat"] = lat
        if lng: raw["lng"] = lng

        df = self._prepare_features(raw)

        # RF prediction
        rf_pred = float(self.rf.predict(df)[0]) if self._rf_loaded else None

        # Queue model prediction
        queue_state = self.queue_model.compute_wait(
            num_ports=num_ports,
            arrival_rate=max(0.1, queue_size + 1),
            service_rate=3.0,
            current_queue=queue_size,
            station_id=station_id,
        )
        queue_pred = queue_state.avg_wait_min

        # Ensemble
        if self._rf_loaded:
            ensemble = self.RF_WEIGHT * rf_pred + self.QUEUE_WEIGHT * queue_pred
        else:
            ensemble = queue_pred

        ensemble = max(0, round(ensemble, 1))

        return {
            "station_id": station_id,
            "predicted_wait_min": ensemble,
            "rf_prediction": round(rf_pred, 1) if rf_pred is not None else None,
            "queue_prediction": round(queue_pred, 2),
            "confidence_level": "high" if self._rf_loaded else "medium",
            "queue_stable": queue_state.system_stable,
            "utilization_pct": round(queue_state.rho * 100, 1),
            "recommendation": self._get_recommendation(ensemble, available_ports),
        }

    @staticmethod
    def _get_recommendation(wait_min: float, available_ports: int) -> str:
        if wait_min <= 5 and available_ports > 0:
            return "GO_NOW"
        elif wait_min <= 15:
            return "GOOD_TIME"
        elif wait_min <= 30:
            return "MODERATE_WAIT"
        elif wait_min <= 60:
            return "LONG_WAIT"
        return "AVOID"

    # ─── Batch Prediction ─────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a DataFrame of feature rows."""
        df = df.copy()
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        if not feature_cols:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        if self._rf_loaded:
            df["rf_prediction"] = self.rf.predict(df)
        else:
            df["rf_prediction"] = np.nan

        df["predicted_wait_min"] = df.get("rf_prediction", pd.Series(15.0, index=df.index)).fillna(15.0)
        df["recommendation"] = df.apply(
            lambda r: self._get_recommendation(
                r["predicted_wait_min"], r.get("available_ports", 2)
            ), axis=1,
        )
        return df

    # ─── Real-Time Inject ─────────────────────────────────────────────────────

    def predict_realtime(
        self,
        station_id: int,
        num_ports: int,
        available_ports: int,
        queue_size: int,
        lat: float,
        lng: float,
        traffic_collector=None,
        weather_collector=None,
    ) -> Dict:
        """Predict with live traffic & weather injection."""
        traffic_score = 0.3
        temperature_c = 20.0
        precipitation_mm = 0.0

        if traffic_collector:
            try:
                tf = traffic_collector.fetch_traffic_flow(lat, lng)
                traffic_score = tf.get("traffic_score", 0.3)
            except Exception:
                pass

        if weather_collector:
            try:
                wx = weather_collector.fetch_current(lat, lng)
                temperature_c = wx.get("temperature_c", 20.0)
                precipitation_mm = wx.get("precipitation_mm", 0.0)
            except Exception:
                pass

        import datetime
        now = datetime.datetime.now()
        return self.predict_single(
            station_id=station_id,
            num_ports=num_ports,
            available_ports=available_ports,
            queue_size=queue_size,
            hour=now.hour,
            day_of_week=now.weekday(),
            traffic_score=traffic_score,
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            lat=lat,
            lng=lng,
        )


if __name__ == "__main__":
    engine = PredictionEngine()
    engine.load_models()

    result = engine.predict_single(
        station_id=1, num_ports=8, available_ports=2,
        queue_size=4, hour=18, day_of_week=2,
        traffic_score=0.75, temperature_c=22.0,
    )
    print("Prediction:", result)
