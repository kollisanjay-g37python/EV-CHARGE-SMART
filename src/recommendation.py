"""
src/recommendation.py
Multi-criteria recommendation engine for EV charging stations.
Scores each station using weighted combination of:
  - Predicted wait time      (lower = better)
  - Distance from user       (closer = better)
  - Port availability        (more = better)
  - Station reliability      (operational = better)
  - Traffic congestion       (lower = better)
  - Power level match        (fast charger preferred)
  - User preferences         (connector type, operator)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import DEFAULT_LOCATION

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User-configurable recommendation preferences."""
    connector_type: Optional[str] = None   # e.g. "CCS", "CHAdeMO", "Type 2"
    min_power_kw: float = 0.0
    max_detour_km: float = 20.0
    priority: str = "balanced"             # "speed" | "distance" | "availability" | "balanced"
    operator_preference: Optional[str] = None

    # Weights (must sum to 1.0)
    weight_wait: float = 0.35
    weight_distance: float = 0.25
    weight_availability: float = 0.20
    weight_reliability: float = 0.10
    weight_traffic: float = 0.10

    def __post_init__(self):
        if self.priority == "speed":
            self.weight_wait = 0.50; self.weight_distance = 0.20
            self.weight_availability = 0.15; self.weight_reliability = 0.10; self.weight_traffic = 0.05
        elif self.priority == "distance":
            self.weight_wait = 0.20; self.weight_distance = 0.50
            self.weight_availability = 0.15; self.weight_reliability = 0.10; self.weight_traffic = 0.05
        elif self.priority == "availability":
            self.weight_wait = 0.20; self.weight_distance = 0.15
            self.weight_availability = 0.50; self.weight_reliability = 0.10; self.weight_traffic = 0.05


@dataclass
class StationScore:
    """Scored and ranked station result."""
    station_id: int
    name: str
    lat: float
    lng: float
    distance_km: float
    predicted_wait_min: float
    available_ports: int
    total_ports: int
    utilization_pct: float
    connector_type: str
    power_kw: float
    operator: str
    status: str
    traffic_score: float
    composite_score: float
    rank: int = 0
    score_breakdown: Dict = field(default_factory=dict)
    routing_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "station_id": self.station_id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "distance_km": round(self.distance_km, 2),
            "predicted_wait_min": round(self.predicted_wait_min, 1),
            "available_ports": self.available_ports,
            "total_ports": self.total_ports,
            "utilization_pct": round(self.utilization_pct, 1),
            "connector_type": self.connector_type,
            "power_kw": self.power_kw,
            "operator": self.operator,
            "status": self.status,
            "traffic_score": round(self.traffic_score, 3),
            "composite_score": round(self.composite_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "routing_url": self.routing_url,
            "action": self._action_label(),
        }

    #def _action_label(self) -> str:
    #    if self.predicted_wait_min <= 5 and self.available_ports > 0:
    #        return "⚡ GO NOW"
    #    elif self.predicted_wait_min <= 15:
    #        return "✅ RECOMMENDED"
    #    elif self.predicted_wait_min <= 30:
    #        return "⏳ MODERATE WAIT"
    #    return "🔴 HIGH WAIT"
    
    def _action_label(self) -> str:
        if self.available_ports <= 0:
            return "❌ FULL"

    # Very low wait time
        elif self.predicted_wait_min < 10:
            return "✅ RECOMMENDED"

    # Moderate wait
        elif self.predicted_wait_min < 30:
            return "⏳ MODERATE WAIT"

    # High congestion
        return "🚫 HIGH WAIT"


class RecommendationEngine:
    """
    Scores and ranks charging stations for a given user location.
    Integrates with PredictionEngine for ML-based wait-time estimates.
    """

    MAX_WAIT_NORMALISE = 60.0    # minutes: maps to score 0
    MAX_DIST_NORMALISE = 20.0    # km: maps to score 0

    def __init__(self, prediction_engine=None):
        self.prediction_engine = prediction_engine

    # ─── Core Scoring ─────────────────────────────────────────────────────────

    def score_station(
        self,
        station: dict,
        user_lat: float,
        user_lng: float,
        prefs: UserPreferences,
        predictions: Optional[Dict] = None,
    ) -> StationScore:
        """Compute composite score for one station."""
        dist = self._haversine(user_lat, user_lng, station["lat"], station["lng"])

        wait = (
            predictions.get("predicted_wait_min", 20.0)
            if predictions else station.get("wait_time_minutes", 20.0)
        )
        avail = station.get("available_ports", 1)
        total = station.get("num_ports", max(avail, 1))
        traffic = station.get("traffic_score", 0.3)
        status = str(station.get("status", "Unknown"))

        # Normalised sub-scores (higher = better)
        s_wait = max(0.0, 1.0 - wait / self.MAX_WAIT_NORMALISE)
        s_dist = max(0.0, 1.0 - dist / self.MAX_DIST_NORMALISE)
        s_avail = avail / max(total, 1)
        s_reliability = 1.0 if "Operational" in status or "Online" in status else 0.3
        s_traffic = max(0.0, 1.0 - traffic)

        breakdown = {
            "wait_score": s_wait,
            "distance_score": s_dist,
            "availability_score": s_avail,
            "reliability_score": s_reliability,
            "traffic_score": s_traffic,
        }

        composite = (
            prefs.weight_wait * s_wait
            + prefs.weight_distance * s_dist
            + prefs.weight_availability * s_avail
            + prefs.weight_reliability * s_reliability
            + prefs.weight_traffic * s_traffic
        )

        # Connector preference bonus
        if prefs.connector_type and prefs.connector_type.lower() in str(station.get("connector_type", "")).lower():
            composite = min(1.0, composite + 0.05)

        # Power bonus for fast chargers
        power = station.get("power_kw", 50.0) or 50.0
        if power >= 150:
            composite = min(1.0, composite + 0.02)

        routing_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&destination={station['lat']},{station['lng']}"
            f"&travelmode=driving"
        )

        return StationScore(
            station_id=station.get("station_id", 0),
            name=str(station.get("name", "Unknown Station")),
            lat=station["lat"],
            lng=station["lng"],
            distance_km=dist,
            predicted_wait_min=wait,
            available_ports=avail,
            total_ports=total,
            utilization_pct=(1 - s_avail) * 100,
            connector_type=str(station.get("connector_type", "Unknown")),
            power_kw=power,
            operator=str(station.get("operator", "")),
            status=status,
            traffic_score=traffic,
            composite_score=composite,
            score_breakdown=breakdown,
            routing_url=routing_url,
        )

    # ─── Rank Network ─────────────────────────────────────────────────────────

    def recommend(
        self,
        stations_df: pd.DataFrame,
        user_lat: float,
        user_lng: float,
        prefs: Optional[UserPreferences] = None,
        top_n: int = 5,
        prediction_df: Optional[pd.DataFrame] = None,
    ) -> List[StationScore]:
        """
        Rank all stations and return top_n recommendations.
        prediction_df: optional DataFrame with 'station_id' and 'predicted_wait_min'.
        """
        if prefs is None:
            prefs = UserPreferences()

        # Build prediction lookup
        pred_lookup: Dict[int, float] = {}
        if prediction_df is not None and not prediction_df.empty:
            for _, row in prediction_df.iterrows():
                pred_lookup[int(row.get("station_id", 0))] = row.get("predicted_wait_min", 20.0)

        scored = []
        for _, station in stations_df.iterrows():
            row = station.to_dict()
            dist = self._haversine(user_lat, user_lng, row.get("lat", 0), row.get("lng", 0))
            if dist > prefs.max_detour_km:
                continue

            sid = int(row.get("station_id", 0))
            preds = {"predicted_wait_min": pred_lookup.get(sid, 20.0)} if sid in pred_lookup else None

            score = self.score_station(row, user_lat, user_lng, prefs, preds)
            scored.append(score)

        # Sort descending by composite score
        scored.sort(key=lambda x: x.composite_score, reverse=True)
        for rank, s in enumerate(scored[:top_n], 1):
            s.rank = rank

        logger.info(f"Recommended top {min(top_n, len(scored))} of {len(scored)} stations")
        return scored[:top_n]

    # ─── Alternative Route Recommendations ───────────────────────────────────

    def recommend_en_route(
        self,
        stations_df: pd.DataFrame,
        route_waypoints: List[Dict],   # [{"lat": ..., "lng": ...}, ...]
        max_detour_km: float = 5.0,
        prefs: Optional[UserPreferences] = None,
        top_n: int = 3,
    ) -> List[StationScore]:
        """Find stations close to a driving route (waypoints)."""
        if prefs is None:
            prefs = UserPreferences(max_detour_km=max_detour_km)

        candidates = []
        for _, station in stations_df.iterrows():
            row = station.to_dict()
            min_dist = min(
                self._haversine(wp["lat"], wp["lng"], row.get("lat", 0), row.get("lng", 0))
                for wp in route_waypoints
            )
            if min_dist <= max_detour_km:
                score = self.score_station(
                    row, route_waypoints[0]["lat"], route_waypoints[0]["lng"], prefs
                )
                score.distance_km = round(min_dist, 2)
                candidates.append(score)

        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        for rank, s in enumerate(candidates[:top_n], 1):
            s.rank = rank
        return candidates[:top_n]

    # ─── Demand Forecast Integration ──────────────────────────────────────────

    def recommend_future(
        self,
        stations_df: pd.DataFrame,
        user_lat: float,
        user_lng: float,
        target_hour: int,
        prefs: Optional[UserPreferences] = None,
    ) -> List[StationScore]:
        """
        Recommend stations for a future arrival time.
        Uses ML model predictions for target_hour.
        """
        if self.prediction_engine is None:
            return self.recommend(stations_df, user_lat, user_lng, prefs)

        import datetime
        now = datetime.datetime.now()
        future_rows = []
        for _, s in stations_df.iterrows():
            row = s.to_dict()
            try:
                pred = self.prediction_engine.predict_single(
                    station_id=int(row.get("station_id", 0)),
                    num_ports=int(row.get("num_ports", 4)),
                    available_ports=int(row.get("available_ports", 2)),
                    queue_size=0,
                    hour=target_hour,
                    day_of_week=now.weekday(),
                    traffic_score=0.4,
                )
                row["wait_time_minutes"] = pred["predicted_wait_min"]
            except Exception:
                pass
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)
        return self.recommend(future_df, user_lat, user_lng, prefs)

    # ─── Geospatial ───────────────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ─── Summary Table ────────────────────────────────────────────────────────

    @staticmethod
    def to_dataframe(scores: List[StationScore]) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in scores])


if __name__ == "__main__":
    # Demo with synthetic stations
    np.random.seed(42)
    n = 20
    stations = pd.DataFrame({
        "station_id": range(n),
        "name": [f"Station {i}" for i in range(n)],
        "lat": 37.77 + np.random.uniform(-0.05, 0.05, n),
        "lng": -122.42 + np.random.uniform(-0.08, 0.08, n),
        "num_ports": np.random.randint(2, 12, n),
        "available_ports": np.random.randint(0, 8, n),
        "traffic_score": np.random.uniform(0.1, 0.9, n),
        "connector_type": np.random.choice(["CCS", "CHAdeMO", "Type 2"], n),
        "power_kw": np.random.choice([7.2, 50, 150, 350], n),
        "status": np.random.choice(["Operational", "Partial", "Offline"], n, p=[0.7, 0.2, 0.1]),
        "operator": np.random.choice(["Tesla", "ChargePoint", "EVgo"], n),
        "wait_time_minutes": np.random.uniform(0, 45, n),
    })

    engine = RecommendationEngine()
    user_lat, user_lng = DEFAULT_LOCATION["lat"], DEFAULT_LOCATION["lng"]
    prefs = UserPreferences(priority="speed", connector_type="CCS")
    results = engine.recommend(stations, user_lat, user_lng, prefs, top_n=5)
    print(RecommendationEngine.to_dataframe(results).to_string(index=False))
