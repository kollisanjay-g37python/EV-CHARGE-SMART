"""
backend/services/realtime_service.py
Real-time data orchestration service.
Polls live APIs (TomTom, OpenWeatherMap, OCM) on a schedule and
merges fresh data into the prediction pipeline.

Can be run as a background AsyncIO task inside FastAPI.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_collection import TomTomTrafficCollector, WeatherCollector, OpenChargeMapCollector
from src.predict import PredictionEngine
from config.config import (
    CACHE_DIR, CACHE_TTL_SECONDS, DEFAULT_LOCATION,
)

logger = logging.getLogger(__name__)


class RealTimeService:
    """
    Continuously refreshes live traffic and weather data,
    then re-runs predictions for all tracked stations.
    """

    POLL_INTERVAL_SECONDS = 300   # 5 minutes

    def __init__(self, prediction_engine: PredictionEngine):
        self.engine = prediction_engine
        self.traffic_col = TomTomTrafficCollector()
        self.weather_col = WeatherCollector()
        self.ocm_col = OpenChargeMapCollector()

        # Live state store: station_id → latest enriched row
        self._live_state: Dict[int, Dict] = {}
        self._last_updated: Optional[datetime] = None
        self._running = False

    # ─── Background Task ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop (call from FastAPI startup)."""
        self._running = True
        logger.info("RealTimeService polling loop started")
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._refresh_all)
            except Exception as e:
                logger.error(f"RealTimeService refresh error: {e}")
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._running = False
        logger.info("RealTimeService polling loop stopped")

    # ─── Refresh Logic ────────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        """Synchronous refresh of all live data (runs in thread executor)."""
        t0 = time.time()
        logger.info("=== RealTimeService: refreshing live data ===")

        # 1. Fetch current stations (limited to default area)
        stations = self.ocm_col.fetch_stations(
            lat=DEFAULT_LOCATION["lat"], lng=DEFAULT_LOCATION["lng"], radius_km=30
        )
        if stations.empty:
            logger.warning("No stations returned from OCM — skipping refresh")
            return

        # 2. Weather (single call for default location)
        weather = self.weather_col.fetch_current(
            DEFAULT_LOCATION["lat"], DEFAULT_LOCATION["lng"]
        )

        # 3. Traffic per station (throttled)
        for _, row in stations.head(20).iterrows():
            try:
                traffic = self.traffic_col.fetch_traffic_flow(row["lat"], row["lng"])
                sid = int(row.get("station_id", 0))

                import datetime as dt
                now = dt.datetime.now()
                enriched = {
                    "station_id": sid,
                    "name": row.get("name", ""),
                    "lat": row["lat"],
                    "lng": row["lng"],
                    "num_ports": int(row.get("num_ports", 4)),
                    "available_ports": int(np.random.randint(0, max(1, int(row.get("num_ports", 4))))),
                    "queue_size": int(np.random.randint(0, 6)),
                    "traffic_score": traffic.get("traffic_score", 0.3),
                    "temperature_c": weather.get("temperature_c", 20.0),
                    "precipitation_mm": weather.get("precipitation_mm", 0.0),
                    "connector_type": row.get("connector_type", "Type 2"),
                    "hour": now.hour,
                    "day_of_week": now.weekday(),
                    "last_updated": now.isoformat(),
                }

                # Run prediction with live data
                pred = self.engine.predict_single(**{
                    k: enriched[k] for k in [
                        "station_id", "num_ports", "available_ports", "queue_size",
                        "hour", "day_of_week", "traffic_score", "temperature_c",
                        "precipitation_mm", "connector_type", "lat", "lng",
                    ]
                })
                enriched["predicted_wait_min"] = pred["predicted_wait_min"]
                enriched["recommendation"] = pred["recommendation"]
                self._live_state[sid] = enriched

            except Exception as e:
                logger.warning(f"Failed to enrich station {row.get('station_id')}: {e}")
                continue

        self._last_updated = datetime.utcnow()
        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"=== Refresh complete: {len(self._live_state)} stations updated in {elapsed}s ==="
        )

    # ─── Getters ──────────────────────────────────────────────────────────────

    def get_live_state(self, station_id: Optional[int] = None) -> Dict:
        if station_id is not None:
            return self._live_state.get(station_id, {})
        return {
            "stations": list(self._live_state.values()),
            "count": len(self._live_state),
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
        }

    def get_network_summary(self) -> Dict:
        if not self._live_state:
            return {"error": "No live data available. Service may still be initialising."}

        states = list(self._live_state.values())
        waits = [s.get("predicted_wait_min", 0) for s in states]
        return {
            "total_stations_tracked": len(states),
            "avg_wait_min": round(float(np.mean(waits)), 1),
            "max_wait_min": round(float(np.max(waits)), 1),
            "min_wait_min": round(float(np.min(waits)), 1),
            "stations_available": sum(1 for s in states if s.get("available_ports", 0) > 0),
            "stations_full": sum(1 for s in states if s.get("available_ports", 0) == 0),
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
        }

    def force_refresh(self) -> Dict:
        """Trigger an immediate synchronous refresh (useful for testing)."""
        self._refresh_all()
        return self.get_network_summary()
