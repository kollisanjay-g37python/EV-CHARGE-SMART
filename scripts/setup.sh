#!/usr/bin/env bash
# scripts/setup.sh
# One-time environment setup for EV Charging Smart System
# Usage: bash scripts/setup.sh

set -euo pipefail

GREEN='\033[0;32m' BLUE='\033[0;34m' YELLOW='\033[1;33m' RED='\033[0;31m' NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ⚡ EV Smart System — Environment Setup   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
log "Checking Python version…"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
  err "Python 3.10+ required. Found: $PYTHON_VERSION"
fi
info "Python $PYTHON_VERSION ✓"

# ── Virtual environment ───────────────────────────────────────────────────────
log "Creating virtual environment…"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  info "Virtual environment created at .venv/"
else
  info "Virtual environment already exists"
fi

source .venv/bin/activate
log "Virtual environment activated"

# ── Python dependencies ───────────────────────────────────────────────────────
log "Installing Python dependencies…"
pip install --upgrade pip -q
pip install -r deployment/requirements.txt -q
info "Python packages installed ✓"

# ── Optional: TensorFlow ──────────────────────────────────────────────────────
read -rp "$(echo -e "${YELLOW}Install TensorFlow for full LSTM support? [y/N]:${NC} ")" TF_ANSWER
if [[ "${TF_ANSWER,,}" == "y" ]]; then
  pip install tensorflow>=2.15.0 -q
  info "TensorFlow installed ✓"
else
  info "Skipping TensorFlow — GBM fallback will be used for LSTM"
fi

# ── Node.js / React ───────────────────────────────────────────────────────────
if command -v node &>/dev/null; then
  NODE_VERSION=$(node --version)
  info "Node.js $NODE_VERSION ✓"
  log "Installing React frontend dependencies…"
  cd frontend && npm install -q && cd ..
  info "React frontend ready ✓"
else
  warn "Node.js not found — React frontend will not be available."
  warn "Install from https://nodejs.org or via: brew install node"
fi

# ── Directory structure ───────────────────────────────────────────────────────
log "Creating data directories…"
mkdir -p data/{raw,processed,real_time_cache} models
info "Directories ready ✓"

# ── .env template ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  log "Creating .env template…"
  cat > .env << 'EOF'
# EV Charging Smart System — Environment Variables
# Copy this file and fill in your API keys

# Open Charge Map (https://openchargemap.org/site/develop/api)
OPENCHARGE_API_KEY=your_key_here

# TomTom Traffic API (https://developer.tomtom.com)
TOMTOM_API_KEY=your_key_here

# OpenWeatherMap (https://openweathermap.org/api)
OPENWEATHER_API_KEY=your_key_here

# Google Maps (optional, for directions)
GOOGLE_MAPS_API_KEY=your_key_here

# Email Alerts (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password

# Alert threshold (minutes)
ALERT_WAIT_THRESHOLD=20

# Backend settings
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
EOF
  info ".env template created — fill in your API keys"
fi

# ── Generate synthetic data ───────────────────────────────────────────────────
log "Generating synthetic training data…"
python3 -c "
import sys; sys.path.insert(0,'.')
from src.data_collection import fetch_charging_stations, generate_synthetic_sessions
stations = fetch_charging_stations()
sessions = generate_synthetic_sessions(stations, days=30)
print(f'  Stations: {len(stations)} | Sessions: {len(sessions):,}')
" 2>/dev/null
info "Synthetic data generated ✓"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Setup Complete! ✓               ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Next steps:                               ║${NC}"
echo -e "${GREEN}║  1. Edit .env with your API keys           ║${NC}"
echo -e "${GREEN}║  2. source .venv/bin/activate              ║${NC}"
echo -e "${GREEN}║  3. python src/train.py                    ║${NC}"
echo -e "${GREEN}║  4. bash scripts/run_all.sh --skip-train   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
