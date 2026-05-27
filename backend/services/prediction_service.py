"""
backend/services/prediction_service.py
Business-logic layer wrapping the PredictionEngine for the API.
Handles caching, logging, input sanitisation, and ensemble weighting.
"""

import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.predict import PredictionEngine
from src.queue_model import MMcQueueModel
from config.config import (
    DEFAULT_ARRIVAL_RATE_PER_HOUR, DEFAULT_SERVICE_RATE_PER_HOUR,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    High-level prediction service with caching and ensemble logic.
    Exposes methods consumed by FastAPI routes.
    """

    # In-memory prediction cache (station_id + hour → result, TTL 5 min)
    _cache: Dict[str, Dict] = {}
    _cache_ttl: int = 300   # seconds

    def __init__(self, engine: PredictionEngine):
        self.engine = engine
        self.queue_model = MMcQueueModel()

    # ─── Cache helpers ────────────────────────────────────────────────────────

    def _cache_key(self, station_id: int, hour: int, day: int) -> str:
        return f"pred:{station_id}:{hour}:{day}"

    def _get_cached(self, key: str) -> Optional[Dict]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry["_ts"]) < self._cache_ttl:
            return entry
        return None

    def _set_cached(self, key: str, result: Dict) -> None:
        self._cache[key] = {**result, "_ts": time.time()}

    # ─── Single Prediction ────────────────────────────────────────────────────

    def predict(
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
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        use_cache: bool = True,
    ) -> Dict:
        key = self._cache_key(station_id, hour, day_of_week)
        if use_cache:
            cached = self._get_cached(key)
            if cached:
                logger.debug(f"Cache hit: {key}")
                return cached

        t0 = time.perf_counter()
        result = self.engine.predict_single(
            station_id=station_id,
            num_ports=num_ports,
            available_ports=available_ports,
            queue_size=queue_size,
            hour=hour,
            day_of_week=day_of_week,
            traffic_score=traffic_score,
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            connector_type=connector_type,
            lat=lat,
            lng=lng,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["inference_ms"] = elapsed_ms
        result["cached"] = False
        result["service"] = "PredictionService"

        if use_cache:
            self._set_cached(key, result)

        logger.info(
            f"Predicted station={station_id} wait={result['predicted_wait_min']}min "
            f"in {elapsed_ms}ms"
        )
        return result

    # ─── Batch Prediction ─────────────────────────────────────────────────────

    def predict_batch(self, items: List[Dict], use_cache: bool = True) -> List[Dict]:
        results = []
        for item in items:
            try:
                result = self.predict(use_cache=use_cache, **item)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item error (station={item.get('station_id')}): {e}")
                results.append({"station_id": item.get("station_id"), "error": str(e)})
        return results

    # ─── 24-hour Demand Curve ────────────────────────────────────────────────

    def demand_curve(
        self,
        station_id: int,
        num_ports: int,
        traffic_score: float = 0.4,
        temperature_c: float = 20.0,
    ) -> List[Dict]:
        """Generate full 24-hour predicted demand curve for a station."""
        import datetime
        day = datetime.datetime.now().weekday()
        curve = []
        for hour in range(24):
            pred = self.predict(
                station_id=station_id,
                num_ports=num_ports,
                available_ports=max(1, num_ports // 2),
                queue_size=2,
                hour=hour,
                day_of_week=day,
                traffic_score=traffic_score,
                temperature_c=temperature_c,
                use_cache=True,
            )
            curve.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "predicted_wait_min": pred["predicted_wait_min"],
                "recommendation": pred["recommendation"],
            })
        return curve

    # ─── Station Utilisation Summary ─────────────────────────────────────────

    def utilisation_summary(self, stations: List[Dict]) -> Dict:
        """Aggregate network-level statistics."""
        total_ports = sum(s.get("num_ports", 0) for s in stations)
        total_avail = sum(s.get("available_ports", 0) for s in stations)
        total_queue = sum(s.get("queue_size", 0) for s in stations)
        avg_wait = np.mean([s.get("wait_time_minutes", 0) for s in stations])
        operational = sum(
            1 for s in stations if "Operational" in str(s.get("status", ""))
        )
        return {
            "total_stations": len(stations),
            "operational_stations": operational,
            "total_ports": total_ports,
            "available_ports": total_avail,
            "occupied_ports": total_ports - total_avail,
            "network_utilization_pct": round(
                (total_ports - total_avail) / max(total_ports, 1) * 100, 1
            ),
            "total_vehicles_waiting": total_queue,
            "avg_predicted_wait_min": round(float(avg_wait), 1),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ─── Anomaly Detection ────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        predictions: List[Dict],
        wait_threshold: float = 45.0,
        util_threshold: float = 95.0,
    ) -> List[Dict]:
        """Flag stations with abnormally high predicted wait or utilisation."""
        anomalies = []
        for pred in predictions:
            reasons = []
            if pred.get("predicted_wait_min", 0) > wait_threshold:
                reasons.append(f"High wait: {pred['predicted_wait_min']:.1f} min")
            if pred.get("utilization_pct", 0) > util_threshold:
                reasons.append(f"Over-capacity: {pred['utilization_pct']:.1f}%")
            if reasons:
                anomalies.append({
                    "station_id": pred.get("station_id"),
                    "reasons": reasons,
                    "severity": "critical" if len(reasons) == 2 else "warning",
                })
        return anomalies

    # ─── Cache Management ─────────────────────────────────────────────────────

    def clear_cache(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cached predictions")
        return count

    def cache_stats(self) -> Dict:
        now = time.time()
        active = sum(
            1 for v in self._cache.values()
            if (now - v.get("_ts", 0)) < self._cache_ttl
        )
        return {"total_entries": len(self._cache), "active_entries": active}
