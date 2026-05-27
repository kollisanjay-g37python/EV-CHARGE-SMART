"""
streamlit_app/utils.py
Helper functions for the Streamlit dashboard.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.predict import PredictionEngine


def load_engine() -> PredictionEngine:
    """Load prediction engine with trained models."""
    engine = PredictionEngine()
    engine.load_models()
    return engine


def generate_demo_stations(n: int = 20) -> pd.DataFrame:
    """Generate synthetic stations for demo/development."""
    np.random.seed(42)
    names = [
        "Tesla Supercharger – Downtown Hub",
        "ChargePoint – Westfield Mall",
        "EVgo – Airport Terminal B",
        "Blink – City Center Plaza",
        "Shell Recharge – Highway 101",
        "Electrify America – Oak Grove",
        "ChargePoint – University Ave",
        "EVgo – Harbor District",
        "Tesla – Financial District",
        "ChargePoint – Caltrain Station",
        "Volta – Ferry Building",
        "EVgo – Fisherman's Wharf",
        "Blink – Golden Gate Park",
        "ChargePoint – Oracle Park",
        "Tesla – Castro District",
        "EVgo – Mission Bay",
        "Electrify America – Stonestown",
        "ChargePoint – SF Zoo",
        "Blink – Twin Peaks",
        "EVgo – Embarcadero",
    ]
    rows = []
    for i in range(min(n, len(names))):
        ports = int(np.random.randint(4, 16))
        avail = int(np.random.randint(0, ports + 1))
        rows.append({
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
                ["Operational", "Operational", "Operational", "Partial", "Offline"],
            ),
            "operator": np.random.choice(["Tesla", "ChargePoint", "EVgo", "Blink"]),
            "traffic_score": round(float(np.random.uniform(0.1, 0.9)), 3),
        })
    return pd.DataFrame(rows)


def get_hourly_demand_data(num_ports: int, base_traffic: float) -> pd.DataFrame:
    """Generate synthetic 24-hour demand curve for a station."""
    hours = list(range(24))
    rows = []
    for h in hours:
        # Rush-hour curve
        if 7 <= h <= 9:
            base = 0.55 + 0.3 * (1 - abs(h - 8) / 2)
        elif 16 <= h <= 19:
            base = 0.7 + 0.25 * (1 - abs(h - 17.5) / 2)
        elif 0 <= h <= 5:
            base = 0.05 + h * 0.01
        elif 22 <= h <= 23:
            base = 0.15
        else:
            base = 0.35 + base_traffic * 0.2

        demand = float(np.clip(base + np.random.normal(0, 0.04), 0, 1))
        predicted = float(np.clip(base + np.random.normal(0, 0.03), 0, 1))
        wait = max(0, demand * 30 + base_traffic * 10 + np.random.normal(0, 2))
        rows.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "demand": round(demand, 3),
            "predicted_demand": round(predicted, 3),
            "predicted_wait_min": round(wait, 1),
            "available_ports": max(0, int(num_ports * (1 - demand))),
        })
    return pd.DataFrame(rows)


def format_wait_badge(wait_min: float) -> str:
    """Return coloured emoji badge for wait time."""
    if wait_min < 5:
        return "🟢 <5 min"
    elif wait_min < 15:
        return f"🟡 {wait_min:.0f} min"
    elif wait_min < 30:
        return f"🟠 {wait_min:.0f} min"
    return f"🔴 {wait_min:.0f} min"


def compute_utilization_color(util_pct: float) -> str:
    if util_pct < 50:
        return "#22c55e"
    elif util_pct < 75:
        return "#f59e0b"
    return "#ef4444"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
