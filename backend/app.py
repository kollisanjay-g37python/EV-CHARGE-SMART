"""
backend/app.py
FastAPI application entry point.
Starts the EV ChargeSmart prediction API server.

Run with:
  uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import API_HOST, API_PORT, API_PREFIX, CORS_ORIGINS, LOG_FORMAT, LOG_LEVEL
from backend.routes import router
from src.predict import PredictionEngine

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─── App Init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EV ChargeSmart API",
    description=(
        "Predicts EV charging station wait times using Random Forest + LSTM ensemble. "
        "Integrates live traffic (TomTom), weather (OpenWeatherMap), "
        "and station data (Open Charge Map / Kaggle)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
engine = PredictionEngine()

@app.on_event("startup")
async def startup():
    logger.info("🔌 EV ChargeSmart API starting up...")
    engine.load_models()
    app.state.engine = engine
    logger.info("✅ Models loaded and ready")

@app.on_event("shutdown")
async def shutdown():
    logger.info("EV ChargeSmart API shutting down")

# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "rf_loaded": engine._rf_loaded,
        "lstm_loaded": engine._lstm_loaded,
    }

# ─── Include Routers ──────────────────────────────────────────────────────────
app.include_router(router, prefix=API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=API_HOST, port=API_PORT, reload=True)
