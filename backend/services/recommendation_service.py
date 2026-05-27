"""
backend/services/recommendation_service.py
Service layer for station recommendation logic.
Wraps RecommendationEngine with business-level features:
  - User history integration
  - Preference persistence
  - Route-aware recommendations
  - A/B-test scoring variants
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.recommendation import RecommendationEngine, StationScore, UserPreferences
from src.predict import PredictionEngine

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    High-level recommendation service used by API routes.
    Adds user-session context and route-aware logic on top of the engine.
    """

    def __init__(self, prediction_engine: PredictionEngine):
        self.engine = RecommendationEngine(prediction_engine=prediction_engine)
        # Simple in-memory user preference store (replace with DB in production)
        self._user_prefs: Dict[str, UserPreferences] = {}

    # ─── Preference Management ────────────────────────────────────────────────

    def save_user_prefs(self, user_id: str, prefs: UserPreferences) -> None:
        self._user_prefs[user_id] = prefs
        logger.info(f"Saved preferences for user {user_id}: priority={prefs.priority}")

    def get_user_prefs(self, user_id: str) -> UserPreferences:
        return self._user_prefs.get(user_id, UserPreferences())

    # ─── Core Recommendation ─────────────────────────────────────────────────

    def recommend(
        self,
        stations_df: pd.DataFrame,
        user_lat: float,
        user_lng: float,
        priority: str = "balanced",
        connector_type: Optional[str] = None,
        max_detour_km: float = 20.0,
        top_n: int = 5,
        user_id: Optional[str] = None,
        prediction_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        # Load saved prefs if user is known
        if user_id and user_id in self._user_prefs:
            prefs = self._user_prefs[user_id]
        else:
            prefs = UserPreferences(
                priority=priority,
                connector_type=connector_type,
                max_detour_km=max_detour_km,
            )

        scores = self.engine.recommend(
            stations_df, user_lat, user_lng, prefs, top_n, prediction_df
        )
        return [s.to_dict() for s in scores]

    # ─── Route-Aware Recommendation ──────────────────────────────────────────

    def recommend_along_route(
        self,
        stations_df: pd.DataFrame,
        waypoints: List[Dict],
        max_detour_km: float = 5.0,
        connector_type: Optional[str] = None,
        top_n: int = 3,
    ) -> List[Dict]:
        if not waypoints:
            return []
        prefs = UserPreferences(
            max_detour_km=max_detour_km,
            connector_type=connector_type,
        )
        scores = self.engine.recommend_en_route(
            stations_df, waypoints, max_detour_km, prefs, top_n
        )
        return [s.to_dict() for s in scores]

    # ─── Future Arrival Recommendation ───────────────────────────────────────

    def recommend_for_arrival(
        self,
        stations_df: pd.DataFrame,
        user_lat: float,
        user_lng: float,
        target_hour: int,
        priority: str = "balanced",
        top_n: int = 5,
    ) -> List[Dict]:
        prefs = UserPreferences(priority=priority)
        scores = self.engine.recommend_future(
            stations_df, user_lat, user_lng, target_hour, prefs
        )
        return [s.to_dict() for s in scores[:top_n]]

    # ─── Diversity Re-ranking ─────────────────────────────────────────────────

    @staticmethod
    def diversify(
        scores: List[Dict], min_distance_km: float = 1.0
    ) -> List[Dict]:
        """
        Remove results that are too close to each other (geospatial diversity).
        Ensures recommendations spread across the network.
        """
        import math
        selected = []
        for candidate in scores:
            too_close = False
            for chosen in selected:
                dlat = candidate["lat"] - chosen["lat"]
                dlng = candidate["lng"] - chosen["lng"]
                dist = math.sqrt(dlat ** 2 + dlng ** 2) * 111
                if dist < min_distance_km:
                    too_close = True
                    break
            if not too_close:
                selected.append(candidate)
        return selected
