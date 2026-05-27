"""
backend/websocket.py
WebSocket endpoint for real-time station status streaming.
Clients connect and receive live updates every N seconds.

Usage in FastAPI:
    from backend.websocket import ws_router
    app.include_router(ws_router)

Client (JavaScript):
    const ws = new WebSocket("ws://localhost:8000/ws/live");
    ws.onmessage = (e) => console.log(JSON.parse(e.data));
"""

import asyncio
import json
import logging
import time
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import RAW_DIR

log = logging.getLogger(__name__)
ws_router = APIRouter()


# ─── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info(f"WS connected — total={len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        log.info(f"WS disconnected — total={len(self.active)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients; remove broken connections."""
        dead = set()
        payload = json.dumps(message)
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.active -= dead

    async def send_personal(self, ws: WebSocket, message: dict):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Live station snapshot ────────────────────────────────────────────────────

def _live_snapshot(station_id: int | None = None) -> dict:
    """
    Generate a real-time station snapshot.
    In production: query OCM / physical sensors.
    Here: controlled randomness seeded on current minute for consistency.
    """
    import pandas as pd
    path = RAW_DIR / "stations.csv"
    rng  = np.random.default_rng(int(time.time()) // 60)   # changes every minute

    if path.exists():
        df = pd.read_csv(path)
        if station_id is not None:
            df = df[df["station_id"] == station_id]
        df = df.head(50)
    else:
        df = pd.DataFrame({
            "station_id":     range(1, 11),
            "name":           [f"Station {i}" for i in range(1, 11)],
            "lat":            37.77 + rng.uniform(-0.05, 0.05, 10),
            "lon":           -122.42 + rng.uniform(-0.05, 0.05, 10),
            "num_connectors": rng.integers(4, 12, 10),
        })

    stations = []
    for _, row in df.iterrows():
        nc     = int(row.get("num_connectors", 4))
        active = int(rng.integers(0, nc + 1))
        queue  = max(0, int(rng.integers(-1, 4)))
        wait   = max(0, round(float(active / max(nc, 1)) * 20 + queue * 3 + rng.normal(0, 1), 1))
        avail  = ("Available" if active < nc * 0.4 else
                  "Moderate"  if active < nc * 0.7 else
                  "Busy"      if active < nc        else "Full")
        stations.append({
            "station_id":      int(row["station_id"]),
            "name":            str(row.get("name", f"Station {row['station_id']}")),
            "lat":             float(row["lat"]),
            "lon":             float(row["lon"]),
            "num_connectors":  nc,
            "active_sessions": active,
            "available_ports": nc - active,
            "queue_length":    queue,
            "wait_min":        wait,
            "availability":    avail,
        })

    return {
        "type":      "station_update",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations":  stations,
    }


# ─── WebSocket endpoints ──────────────────────────────────────────────────────

@ws_router.websocket("/ws/live")
async def ws_live_all(websocket: WebSocket):
    """
    Stream live updates for ALL stations every 15 seconds.
    Client receives: { type, timestamp, stations: [...] }
    """
    await manager.connect(websocket)
    try:
        # Send initial snapshot immediately
        await manager.send_personal(websocket, _live_snapshot())

        while True:
            await asyncio.sleep(15)
            if websocket not in manager.active:
                break
            await manager.send_personal(websocket, _live_snapshot())

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WS error: {e}")
        manager.disconnect(websocket)


@ws_router.websocket("/ws/station/{station_id}")
async def ws_live_station(websocket: WebSocket, station_id: int):
    """
    Stream live updates for a SINGLE station every 10 seconds.
    Client receives: { type, timestamp, stations: [single_station] }
    """
    await manager.connect(websocket)
    try:
        await manager.send_personal(websocket, _live_snapshot(station_id))
        while True:
            await asyncio.sleep(10)
            if websocket not in manager.active:
                break
            await manager.send_personal(websocket, _live_snapshot(station_id))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WS station/{station_id} error: {e}")
        manager.disconnect(websocket)


@ws_router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Stream wait-time alert events.
    Client receives alerts only when wait > threshold.
    """
    from config.config import ALERT_WAIT_THRESHOLD
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(20)
            if websocket not in manager.active:
                break
            snapshot = _live_snapshot()
            alerts   = [
                {**s, "alert": True, "message": f"⚠️ {s['name']} — {s['wait_min']} min wait"}
                for s in snapshot["stations"]
                if s["wait_min"] >= ALERT_WAIT_THRESHOLD
            ]
            if alerts:
                await manager.send_personal(websocket, {
                    "type":      "alert",
                    "timestamp": snapshot["timestamp"],
                    "alerts":    alerts,
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WS alerts error: {e}")
        manager.disconnect(websocket)
