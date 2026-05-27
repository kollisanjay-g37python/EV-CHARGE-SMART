"""
backend/routes.py
All API route definitions for EV ChargeSmart.

Endpoints:
  POST /api/v1/predict              → single station wait-time prediction
  POST /api/v1/predict/batch        → batch predictions for multiple stations
  GET  /api/v1/stations             → list stations with live status
  GET  /api/v1/stations/{id}        → single station detail
  POST /api/v1/recommend            → ranked station recommendations
  GET  /api/v1/queue/{station_id}   → M/M/c queue analysis
  GET  /api/v1/forecast/{station_id}→ 12-hour demand forecast
  GET  /api/v1/metrics              → model performance metrics
  POST /api/v1/realtime             → real-time prediction with live APIs
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, validator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import PredictionEngine
from src.recommendation import RecommendationEngine, UserPreferences
from src.queue_model import MMcQueueModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    station_id: int = Field(..., example=1)
    num_ports: int = Field(..., ge=1, le=100, example=8)
    available_ports: int = Field(..., ge=0, le=100, example=3)
    queue_size: int = Field(..., ge=0, le=200, example=4)
    hour: int = Field(..., ge=0, le=23, example=18)
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="0=Mon … 6=Sun")
    traffic_score: float = Field(..., ge=0.0, le=1.0, example=0.72)
    temperature_c: float = Field(20.0, example=22.5)
    precipitation_mm: float = Field(0.0, ge=0.0, example=0.0)
    connector_type: str = Field("Type 2", example="CCS")
    lat: Optional[float] = Field(None, example=37.7749)
    lng: Optional[float] = Field(None, example=-122.4194)

    @validator("available_ports")
    def ports_not_exceed_total(cls, v, values):
        if "num_ports" in values and v > values["num_ports"]:
            raise ValueError("available_ports cannot exceed num_ports")
        return v


class PredictResponse(BaseModel):
    station_id: int
    predicted_wait_min: float
    rf_prediction: Optional[float]
    queue_prediction: float
    confidence_level: str
    queue_stable: bool
    utilization_pct: float
    recommendation: str
    timestamp: str


class BatchPredictItem(BaseModel):
    station_id: int
    num_ports: int
    available_ports: int
    queue_size: int
    hour: int
    day_of_week: int
    traffic_score: float
    temperature_c: float = 20.0


class RecommendRequest(BaseModel):
    user_lat: float = Field(..., example=37.7749)
    user_lng: float = Field(..., example=-122.4194)
    priority: str = Field("balanced", example="speed",
                          description="speed | distance | availability | balanced")
    connector_type: Optional[str] = Field(None, example="CCS")
    max_detour_km: float = Field(20.0, ge=0.5, le=100.0)
    top_n: int = Field(5, ge=1, le=20)
    target_hour: Optional[int] = Field(None, ge=0, le=23,
                                       description="Future arrival hour for forecast")


class StationStatus(BaseModel):
    station_id: int
    name: str
    lat: float
    lng: float
    connector_type: str
    num_ports: int
    available_ports: int
    queue_size: int
    predicted_wait_min: float
    utilization_pct: float
    status: str
    traffic_score: float


class QueueAnalysisResponse(BaseModel):
    station_id: int
    station_name: str
    num_ports: int
    arrival_rate: float
    service_rate: float
    rho: float
    erlang_c: float
    avg_wait_min: float
    avg_queue_length: float
    throughput_per_hr: float
    system_stable: bool
    utilization_pct: float


# ─── Dependency: get engine from app state ────────────────────────────────────

def get_engine(request: Request) -> PredictionEngine:
    return request.app.state.engine


# ─── Synthetic station store (replace with DB in production) ──────────────────

def _mock_stations(n: int = 20) -> List[dict]:
    np.random.seed(42)
    names = [
        "Tesla Supercharger – Downtown", "ChargePoint – Westfield Mall",
        "EVgo – Airport Terminal B", "Blink – City Center Plaza",
        "Shell Recharge – Highway 101", "Electrify America – Oak Grove",
        "ChargePoint – University Ave", "EVgo – Harbor District",
        "Tesla – Financial District", "ChargePoint – Caltrain Station",
        "Volta – Ferry Building", "EVgo – Fisherman's Wharf",
        "Blink – Golden Gate Park", "ChargePoint – Oracle Park",
        "Tesla – Castro District", "EVgo – Mission Bay",
        "Electrify America – Stonestown", "ChargePoint – SF Zoo",
        "Blink – Twin Peaks", "EVgo – Embarcadero",
    ]
    stations = []
    for i in range(min(n, len(names))):
        ports = int(np.random.randint(4, 16))
        avail = int(np.random.randint(0, ports))
        stations.append({
            "station_id": i + 1,
            "name": names[i],
            "lat": round(37.7749 + np.random.uniform(-0.06, 0.06), 5),
            "lng": round(-122.4194 + np.random.uniform(-0.10, 0.10), 5),
            "num_ports": ports,
            "available_ports": avail,
            "queue_size": int(np.random.randint(0, 7)),
            "connector_type": np.random.choice(["CCS", "CHAdeMO", "Type 2", "Tesla"]),
            "power_kw": float(np.random.choice([7.2, 50.0, 150.0, 350.0])),
            "status": np.random.choice(
                ["Operational", "Operational", "Operational", "Partial", "Offline"]
            ),
            "operator": np.random.choice(["Tesla", "ChargePoint", "EVgo", "Blink"]),
            "traffic_score": round(float(np.random.uniform(0.1, 0.9)), 3),
            "wait_time_minutes": round(float(np.random.uniform(0, 45)), 1),
        })
    return stations


_STATIONS = _mock_stations()


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict wait time for a single station",
    tags=["Prediction"],
)
async def predict_single(
    body: PredictRequest,
    engine: PredictionEngine = Depends(get_engine),
):
    try:
        result = engine.predict_single(
            station_id=body.station_id,
            num_ports=body.num_ports,
            available_ports=body.available_ports,
            queue_size=body.queue_size,
            hour=body.hour,
            day_of_week=body.day_of_week,
            traffic_score=body.traffic_score,
            temperature_c=body.temperature_c,
            precipitation_mm=body.precipitation_mm,
            connector_type=body.connector_type,
            lat=body.lat,
            lng=body.lng,
        )
        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/batch",
    summary="Batch predictions for multiple stations",
    tags=["Prediction"],
)
async def predict_batch(
    items: List[BatchPredictItem],
    engine: PredictionEngine = Depends(get_engine),
):
    results = []
    for item in items:
        try:
            result = engine.predict_single(
                station_id=item.station_id,
                num_ports=item.num_ports,
                available_ports=item.available_ports,
                queue_size=item.queue_size,
                hour=item.hour,
                day_of_week=item.day_of_week,
                traffic_score=item.traffic_score,
                temperature_c=item.temperature_c,
            )
            results.append(result)
        except Exception as e:
            results.append({"station_id": item.station_id, "error": str(e)})
    return {"predictions": results, "count": len(results)}


@router.get(
    "/stations",
    summary="List all stations with current status",
    tags=["Stations"],
)
async def list_stations(
    lat: Optional[float] = Query(None, description="User latitude for distance sort"),
    lng: Optional[float] = Query(None, description="User longitude for distance sort"),
    status: Optional[str] = Query(None, description="Filter by status: Operational|Partial|Offline"),
    connector: Optional[str] = Query(None, description="Filter by connector type"),
    limit: int = Query(20, ge=1, le=100),
):
    stations = _STATIONS.copy()

    if status:
        stations = [s for s in stations if status.lower() in s["status"].lower()]
    if connector:
        stations = [s for s in stations if connector.lower() in s["connector_type"].lower()]

    if lat is not None and lng is not None:
        import math
        for s in stations:
            s["distance_km"] = round(
                6371 * 2 * math.asin(
                    math.sqrt(
                        math.sin(math.radians((s["lat"] - lat) / 2)) ** 2
                        + math.cos(math.radians(lat))
                        * math.cos(math.radians(s["lat"]))
                        * math.sin(math.radians((s["lng"] - lng) / 2)) ** 2
                    )
                ), 2
            )
        stations.sort(key=lambda s: s.get("distance_km", 999))

    return {
        "stations": stations[:limit],
        "total": len(stations),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/stations/{station_id}",
    summary="Get single station details",
    tags=["Stations"],
)
async def get_station(station_id: int, engine: PredictionEngine = Depends(get_engine)):
    station = next((s for s in _STATIONS if s["station_id"] == station_id), None)
    if not station:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
    import datetime as dt
    hour = dt.datetime.now().hour
    pred = engine.predict_single(
        station_id=station_id,
        num_ports=station["num_ports"],
        available_ports=station["available_ports"],
        queue_size=station["queue_size"],
        hour=hour,
        day_of_week=dt.datetime.now().weekday(),
        traffic_score=station["traffic_score"],
    )
    return {**station, "prediction": pred}


@router.post(
    "/recommend",
    summary="Get ranked station recommendations",
    tags=["Recommendation"],
)
async def recommend(
    body: RecommendRequest,
    engine: PredictionEngine = Depends(get_engine),
):
    stations_df = pd.DataFrame(_STATIONS)
    prefs = UserPreferences(
        priority=body.priority,
        connector_type=body.connector_type,
        max_detour_km=body.max_detour_km,
    )
    rec_engine = RecommendationEngine(prediction_engine=engine)

    if body.target_hour is not None:
        results = rec_engine.recommend_future(
            stations_df, body.user_lat, body.user_lng, body.target_hour, prefs
        )
    else:
        results = rec_engine.recommend(
            stations_df, body.user_lat, body.user_lng, prefs, top_n=body.top_n
        )

    return {
        "recommendations": [r.to_dict() for r in results],
        "count": len(results),
        "user_location": {"lat": body.user_lat, "lng": body.user_lng},
        "priority": body.priority,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/queue/{station_id}",
    response_model=QueueAnalysisResponse,
    summary="M/M/c queue analysis for a station",
    tags=["Queue Model"],
)
async def queue_analysis(
    station_id: int,
    arrival_rate: float = Query(8.0, ge=0.1, description="Vehicles per hour"),
    service_rate: float = Query(3.0, ge=0.1, description="Sessions per port per hour"),
):
    station = next((s for s in _STATIONS if s["station_id"] == station_id), None)
    if not station:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    qm = MMcQueueModel()
    state = qm.compute_wait(
        num_ports=station["num_ports"],
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        current_queue=station["queue_size"],
        station_id=station_id,
        station_name=station["name"],
    )
    return state.to_dict()


@router.get(
    "/forecast/{station_id}",
    summary="12-hour demand forecast for a station",
    tags=["Forecast"],
)
async def forecast(
    station_id: int,
    engine: PredictionEngine = Depends(get_engine),
):
    import datetime as dt
    station = next((s for s in _STATIONS if s["station_id"] == station_id), None)
    if not station:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    now = dt.datetime.now()
    forecast_points = []
    for offset in range(12):
        future_hour = (now.hour + offset) % 24
        pred = engine.predict_single(
            station_id=station_id,
            num_ports=station["num_ports"],
            available_ports=station["available_ports"],
            queue_size=station["queue_size"],
            hour=future_hour,
            day_of_week=now.weekday(),
            traffic_score=station["traffic_score"],
        )
        forecast_points.append({
            "hour": future_hour,
            "label": f"{future_hour:02d}:00",
            "offset_hours": offset,
            "predicted_wait_min": pred["predicted_wait_min"],
            "recommendation": pred["recommendation"],
        })

    return {
        "station_id": station_id,
        "station_name": station["name"],
        "forecast": forecast_points,
        "generated_at": now.isoformat(),
    }


@router.get(
    "/metrics",
    summary="Model performance metrics",
    tags=["Monitoring"],
)
async def model_metrics():
    import json
    metrics_path = Path(__file__).resolve().parent.parent / "models" / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {
        "message": "No training metrics found. Run src/train.py first.",
        "random_forest": {"rmse_min": "N/A", "mae_min": "N/A", "r2": "N/A"},
        "lstm": {"rmse_min": "N/A", "mae_min": "N/A"},
    }


@router.get(
    "/capacity-plan",
    summary="Capacity planning: recommend minimum ports for target wait",
    tags=["Queue Model"],
)
async def capacity_plan(
    arrival_rate: float = Query(10.0, ge=0.1),
    service_rate: float = Query(3.0, ge=0.1),
    target_wait_min: float = Query(15.0, ge=1.0),
):
    qm = MMcQueueModel()
    return qm.recommend_capacity(arrival_rate, service_rate, target_wait_min)
