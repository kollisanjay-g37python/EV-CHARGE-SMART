"""
src/data_collection.py
Collects data from all external sources:
  - Open Charge Map API (station metadata)
  - TomTom Traffic API (real-time + historical traffic)
  - OpenWeatherMap API (weather features)
  - Kaggle CSV datasets (charging sessions, energy demand)

Datasets:
  Stations  : https://openchargemap.org/site/develop/api
             https://www.kaggle.com/datasets/risheepanchal/global-ev-charging-stations-dataset
  Traffic   : https://developer.tomtom.com/traffic-api
  Weather   : https://openweathermap.org/api
  Sessions  : https://www.kaggle.com/datasets/datasetengineer/ev-charging-load-dataset-and-optimal-routing
  Energy    : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
  Demand    : https://www.kaggle.com/datasets/salader/ev-demand-prediction
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    CACHE_DIR, CACHE_TTL_SECONDS, DEFAULT_LOCATION, DEFAULT_RADIUS_KM,
    GOOGLE_DIRECTIONS_URL, GOOGLE_MAPS_API_KEY, KAGGLE_DEMAND_CSV,
    KAGGLE_ENERGY_CSV, KAGGLE_LOAD_CSV, KAGGLE_STATIONS_CSV,
    MAX_STATIONS_PER_CALL, OCM_BASE_URL, OPEN_CHARGE_MAP_API_KEY,
    OPENWEATHER_API_KEY, OWM_BASE_URL, RAW_DIR, TOMTOM_API_KEY,
    TOMTOM_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[dict]:
    path = _get_cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            with open(path) as f:
                logger.debug(f"Cache hit: {key}")
                return json.load(f)
    return None


def _save_cache(key: str, data: dict) -> None:
    with open(_get_cache_path(key), "w") as f:
        json.dump(data, f)


def _safe_get(url: str, params: dict, source: str) -> Optional[dict]:
    """HTTP GET with retry logic and error handling."""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[{source}] HTTP error ({e.response.status_code}): {e}")
            break
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{source}] Connection error, attempt {attempt + 1}/3")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"[{source}] Unexpected error: {e}")
            break
    return None


# ─── 1. Open Charge Map ───────────────────────────────────────────────────────

class OpenChargeMapCollector:
    """
    Fetches EV charging station data from Open Charge Map API.
    Docs: https://openchargemap.org/site/develop/api
    """

    def fetch_stations(
        self,
        lat: float = DEFAULT_LOCATION["lat"],
        lng: float = DEFAULT_LOCATION["lng"],
        radius_km: float = DEFAULT_RADIUS_KM,
        max_results: int = MAX_STATIONS_PER_CALL,
    ) -> pd.DataFrame:
        cache_key = f"ocm_{lat:.3f}_{lng:.3f}_{radius_km}"
        cached = _load_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        logger.info(f"Fetching stations from Open Charge Map (lat={lat}, lng={lng}, r={radius_km}km)")
        params = {
            "key": OPEN_CHARGE_MAP_API_KEY,
            "latitude": lat,
            "longitude": lng,
            "distance": radius_km,
            "distanceunit": "KM",
            "maxresults": max_results,
            "compact": True,
            "verbose": False,
            "output": "json",
        }
        data = _safe_get(f"{OCM_BASE_URL}/poi/", params, "OpenChargeMap")
        if not data:
            logger.warning("OCM API failed — loading from Kaggle CSV fallback")
            return self._load_kaggle_fallback()

        records = []
        for station in data:
            addr = station.get("AddressInfo", {})
            connections = station.get("Connections", [{}])
            conn = connections[0] if connections else {}
            records.append({
                "station_id": station.get("ID"),
                "name": addr.get("Title", "Unknown"),
                "lat": addr.get("Latitude"),
                "lng": addr.get("Longitude"),
                "address": addr.get("AddressLine1", ""),
                "city": addr.get("Town", ""),
                "country": addr.get("Country", {}).get("ISOCode", ""),
                "num_ports": len(connections),
                "connector_type": conn.get("ConnectionType", {}).get("Title", "Unknown"),
                "power_kw": conn.get("PowerKW"),
                "status": station.get("StatusType", {}).get("Title", "Unknown") if station.get("StatusType") else "Unknown",
                "usage_type": station.get("UsageType", {}).get("Title", "Public") if station.get("UsageType") else "Public",
                "date_created": station.get("DateCreated"),
                "operator": station.get("OperatorInfo", {}).get("Title", "") if station.get("OperatorInfo") else "",
            })

        df = pd.DataFrame(records)
        _save_cache(cache_key, df.to_dict("records"))
        logger.info(f"Fetched {len(df)} stations from OCM")
        return df

    def _load_kaggle_fallback(self) -> pd.DataFrame:
        """Load from Kaggle global EV stations CSV when API unavailable."""
        if KAGGLE_STATIONS_CSV.exists():
            logger.info(f"Loading Kaggle station fallback: {KAGGLE_STATIONS_CSV}")
            df = pd.read_csv(KAGGLE_STATIONS_CSV)
            # Standardise common column names
            col_map = {
                "Latitude": "lat", "Longitude": "lng",
                "Name": "name", "ID": "station_id",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            return df
        logger.error("Kaggle stations CSV not found. Download from: "
                     "https://www.kaggle.com/datasets/risheepanchal/global-ev-charging-stations-dataset")
        return pd.DataFrame()


# ─── 2. TomTom Traffic ────────────────────────────────────────────────────────

class TomTomTrafficCollector:
    """
    Fetches traffic flow data from TomTom Traffic API.
    Docs: https://developer.tomtom.com/traffic-api
    """

    def fetch_traffic_flow(self, lat: float, lng: float) -> Dict:
        cache_key = f"traffic_{lat:.4f}_{lng:.4f}"
        cached = _load_cache(cache_key)
        if cached:
            return cached

        logger.info(f"Fetching TomTom traffic at ({lat}, {lng})")
        url = f"{TOMTOM_BASE_URL}/flowSegmentData/relative0/10/json"
        params = {
            "key": TOMTOM_API_KEY,
            "point": f"{lat},{lng}",
        }
        data = _safe_get(url, params, "TomTom") or {}
        flow = data.get("flowSegmentData", {})

        result = {
            "lat": lat, "lng": lng,
            "current_speed_kmh": flow.get("currentSpeed", 0),
            "free_flow_speed_kmh": flow.get("freeFlowSpeed", 50),
            "confidence": flow.get("confidence", 0),
            "traffic_score": self._compute_score(flow),
            "timestamp": datetime.utcnow().isoformat(),
        }
        _save_cache(cache_key, result)
        return result

    def fetch_bulk_traffic(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch traffic for all stations and merge."""
        records = []
        for _, row in stations_df.iterrows():
            tf = self.fetch_traffic_flow(row["lat"], row["lng"])
            records.append({"station_id": row["station_id"], **tf})
            time.sleep(0.1)   # rate limit courtesy
        return pd.DataFrame(records)

    def _compute_score(self, flow: dict) -> float:
        """Normalise traffic to 0 (free flow) – 1 (jam)."""
        current = flow.get("currentSpeed", 50)
        free = flow.get("freeFlowSpeed", 50)
        if free == 0:
            return 0.0
        ratio = current / free
        return round(max(0.0, min(1.0, 1.0 - ratio)), 4)

    def fetch_historical_traffic(
        self, lat: float, lng: float, days_back: int = 30
    ) -> pd.DataFrame:
        """
        Build synthetic historical traffic using TomTom patterns.
        In production replace with TomTom O/D Matrix or historical flow API.
        """
        logger.info(f"Generating historical traffic proxy for {days_back} days")
        records = []
        base_date = datetime.utcnow() - timedelta(days=days_back)
        for day_offset in range(days_back):
            dt = base_date + timedelta(days=day_offset)
            for hour in range(24):
                # Rush hours model
                if 7 <= hour <= 9 or 16 <= hour <= 19:
                    score = 0.7 + 0.2 * (1 - abs(hour - 17) / 5)
                elif 0 <= hour <= 5:
                    score = 0.05
                else:
                    score = 0.35
                records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": dt.weekday(),
                    "traffic_score": round(score + 0.05 * (0.5 - import_random()), 4),
                })
        return pd.DataFrame(records)


def import_random():
    import random
    return random.random()


# ─── 3. OpenWeatherMap ────────────────────────────────────────────────────────

class WeatherCollector:
    """
    Fetches current & forecast weather from OpenWeatherMap.
    Docs: https://openweathermap.org/api
    """

    def fetch_current(self, lat: float, lng: float) -> Dict:
        cache_key = f"weather_{lat:.3f}_{lng:.3f}"
        cached = _load_cache(cache_key)
        if cached:
            return cached

        logger.info(f"Fetching weather at ({lat}, {lng})")
        params = {
            "lat": lat, "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }
        data = _safe_get(f"{OWM_BASE_URL}/weather", params, "OpenWeatherMap") or {}
        result = {
            "temperature_c": data.get("main", {}).get("temp", 20.0),
            "feels_like_c": data.get("main", {}).get("feels_like", 20.0),
            "humidity_pct": data.get("main", {}).get("humidity", 50),
            "precipitation_mm": data.get("rain", {}).get("1h", 0.0),
            "wind_speed_ms": data.get("wind", {}).get("speed", 0.0),
            "weather_main": data.get("weather", [{}])[0].get("main", "Clear"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        _save_cache(cache_key, result)
        return result

    def fetch_forecast(self, lat: float, lng: float) -> pd.DataFrame:
        """5-day / 3-hour forecast."""
        params = {
            "lat": lat, "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }
        data = _safe_get(f"{OWM_BASE_URL}/forecast", params, "OpenWeatherMap") or {}
        rows = []
        for item in data.get("list", []):
            rows.append({
                "datetime": item.get("dt_txt"),
                "temperature_c": item["main"]["temp"],
                "precipitation_mm": item.get("rain", {}).get("3h", 0.0),
                "weather_main": item["weather"][0]["main"],
            })
        return pd.DataFrame(rows)


# ─── 4. Kaggle Dataset Loaders ────────────────────────────────────────────────

class KaggleDatasetLoader:
    """
    Loads and standardises Kaggle CSV datasets.
    Download instructions provided if files are missing.
    """

    def load_charging_sessions(self) -> pd.DataFrame:
        """
        EV Charging Load Dataset.
        Source: https://www.kaggle.com/datasets/datasetengineer/ev-charging-load-dataset-and-optimal-routing
        """
        if not KAGGLE_LOAD_CSV.exists():
            logger.warning(
                "Charging session dataset not found.\n"
                "Download from: https://www.kaggle.com/datasets/datasetengineer/"
                "ev-charging-load-dataset-and-optimal-routing\n"
                f"Place CSV at: {KAGGLE_LOAD_CSV}"
            )
            return self._generate_synthetic_sessions()

        logger.info(f"Loading charging sessions: {KAGGLE_LOAD_CSV}")
        df = pd.read_csv(KAGGLE_LOAD_CSV, parse_dates=True)
        return df

    def load_hourly_energy(self) -> pd.DataFrame:
        """
        Hourly Energy Consumption (demand proxy).
        Source: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
        """
        if not KAGGLE_ENERGY_CSV.exists():
            logger.warning(
                "Hourly energy CSV not found.\n"
                "Download from: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption\n"
                f"Place CSV at: {KAGGLE_ENERGY_CSV}"
            )
            return pd.DataFrame()

        logger.info(f"Loading hourly energy: {KAGGLE_ENERGY_CSV}")
        df = pd.read_csv(KAGGLE_ENERGY_CSV, parse_dates=["Datetime"], index_col="Datetime")
        return df

    def load_ev_demand(self) -> pd.DataFrame:
        """
        EV Demand Prediction Dataset.
        Source: https://www.kaggle.com/datasets/salader/ev-demand-prediction
        """
        if not KAGGLE_DEMAND_CSV.exists():
            logger.warning(
                "EV demand CSV not found.\n"
                "Download from: https://www.kaggle.com/datasets/salader/ev-demand-prediction\n"
                f"Place CSV at: {KAGGLE_DEMAND_CSV}"
            )
            return pd.DataFrame()

        logger.info(f"Loading EV demand: {KAGGLE_DEMAND_CSV}")
        return pd.read_csv(KAGGLE_DEMAND_CSV, parse_dates=True)

    def _generate_synthetic_sessions(self, n: int = 50_000) -> pd.DataFrame:
        """Generate realistic synthetic charging session data for development."""
        import numpy as np
        import random
        logger.info(f"Generating {n} synthetic charging sessions for development")
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="10min")
        hours = dates.hour
        # Demand curve: low at night, peaks at morning and evening
        demand_weights = (
            0.1 + 0.5 * np.exp(-((hours - 8) ** 2) / 8)
            + 0.7 * np.exp(-((hours - 18) ** 2) / 6)
        )
        demand_weights /= demand_weights.max()

        df = pd.DataFrame({
            "session_id": range(n),
            "station_id": rng.integers(1, 51, n),
            "start_time": dates,
            "session_duration_min": np.clip(rng.normal(35, 18, n), 5, 120),
            "energy_kwh": np.clip(rng.normal(22, 10, n), 1, 80),
            "demand_weight": demand_weights,
            "queue_size": rng.integers(0, 8, n),
            "available_ports": rng.integers(0, 8, n),
            "traffic_score": rng.uniform(0.05, 0.95, n),
            "temperature_c": rng.normal(18, 8, n),
            "precipitation_mm": rng.exponential(0.5, n),
            "connector_type": rng.choice(
                ["Type 2", "CCS", "CHAdeMO", "Tesla", "J1772"], n
            ),
        })
        df["wait_time_minutes"] = np.clip(
            df["queue_size"] * rng.uniform(8, 15, n)
            + df["traffic_score"] * 10
            - df["available_ports"] * 3,
            0, 90,
        ).round(1)
        return df


# ─── Master Collector ─────────────────────────────────────────────────────────

class DataCollector:
    """Orchestrates all data collection and saves raw CSVs."""

    def __init__(self):
        self.ocm = OpenChargeMapCollector()
        self.traffic = TomTomTrafficCollector()
        self.weather = WeatherCollector()
        self.kaggle = KaggleDatasetLoader()
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def collect_all(
        self,
        lat: float = DEFAULT_LOCATION["lat"],
        lng: float = DEFAULT_LOCATION["lng"],
    ) -> Dict[str, pd.DataFrame]:
        logger.info("=== Starting full data collection ===")
        results = {}

        # Stations
        stations = self.ocm.fetch_stations(lat, lng)
        stations.to_csv(RAW_DIR / "stations_raw.csv", index=False)
        results["stations"] = stations

        # Sessions (Kaggle or synthetic)
        sessions = self.kaggle.load_charging_sessions()
        sessions.to_csv(RAW_DIR / "sessions_raw.csv", index=False)
        results["sessions"] = sessions

        # Energy proxy
        energy = self.kaggle.load_hourly_energy()
        if not energy.empty:
            energy.to_csv(RAW_DIR / "energy_raw.csv")
        results["energy"] = energy

        # EV demand
        demand = self.kaggle.load_ev_demand()
        if not demand.empty:
            demand.to_csv(RAW_DIR / "demand_raw.csv", index=False)
        results["demand"] = demand

        # Traffic for first 20 stations
        if not stations.empty:
            traffic_df = self.traffic.fetch_bulk_traffic(stations.head(20))
            traffic_df.to_csv(RAW_DIR / "traffic_raw.csv", index=False)
            results["traffic"] = traffic_df

        # Weather at default location
        weather = self.weather.fetch_current(lat, lng)
        pd.DataFrame([weather]).to_csv(RAW_DIR / "weather_raw.csv", index=False)
        results["weather"] = pd.DataFrame([weather])

        logger.info(f"=== Collection complete. Saved to {RAW_DIR} ===")
        return results


if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all()
    for name, df in data.items():
        print(f"{name}: {df.shape if isinstance(df, pd.DataFrame) else 'N/A'}")
