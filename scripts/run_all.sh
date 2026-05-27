#!/usr/bin/env bash
# scripts/run_all.sh
# Master launcher — runs the full EV Charging Smart System stack
# Usage: bash scripts/run_all.sh [--skip-train] [--demo]

set -euo pipefail

GREEN='\033[0;32m' BLUE='\033[0;34m' YELLOW='\033[1;33m' RED='\033[0;31m' NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SKIP_TRAIN=false
DEMO_MODE=false

for arg in "$@"; do
  case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    --demo)       DEMO_MODE=true  ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ⚡ EV Charging Smart System — Full Stack   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Environment check ──────────────────────────────────────────────────────
log "Checking Python environment…"
python3 -c "import numpy, pandas, sklearn, fastapi, streamlit" 2>/dev/null \
  || { warn "Missing deps — installing…"; pip install -r deployment/requirements.txt -q; }
info "Python dependencies OK"

# ── 2. Training pipeline ──────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
  log "Running training pipeline…"
  python3 src/train.py
  log "Generating evaluation charts…"
  python3 src/evaluate.py
  log "Running EDA notebooks…"
  python3 notebooks/eda.py
  python3 notebooks/time_series_analysis.py
else
  warn "Skipping training (--skip-train flag set)"
fi

# ── 3. Tests ──────────────────────────────────────────────────────────────────
log "Running test suite…"
if command -v pytest &>/dev/null; then
  pytest tests/ -v --tb=short -q 2>&1 | tail -20
else
  warn "pytest not found — skipping tests"
fi

# ── 4. FastAPI backend ────────────────────────────────────────────────────────
log "Starting FastAPI backend on port 8000…"
uvicorn backend.app:app --host 0.0.0.0 --port 8000 \
  --reload --log-level warning &
BACKEND_PID=$!
info "Backend PID: $BACKEND_PID"

# Wait for API to be ready
for i in {1..10}; do
  sleep 1
  curl -sf http://localhost:8000/health &>/dev/null && break
  [ $i -eq 10 ] && warn "Backend health check timed out"
done
info "API ready → http://localhost:8000/docs"

# ── 5. Streamlit dashboard ────────────────────────────────────────────────────
log "Starting Streamlit dashboard on port 8501…"
streamlit run streamlit_app/app.py \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!
info "Dashboard PID: $STREAMLIT_PID"
info "Dashboard → http://localhost:8501"

# ── 6. React frontend (if Node available) ─────────────────────────────────────
if command -v node &>/dev/null && [ "$DEMO_MODE" = false ]; then
  log "Starting React frontend on port 3000…"
  cd frontend
  [ -d "node_modules" ] || npm install -q
  REACT_APP_API_URL=http://localhost:8000/api/v1 npm start &
  REACT_PID=$!
  cd ..
  info "React UI → http://localhost:3000"
else
  warn "Node.js not found or demo mode — skipping React frontend"
  REACT_PID=""
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Services Running                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  FastAPI    → http://localhost:8000/docs     ║${NC}"
echo -e "${GREEN}║  Streamlit  → http://localhost:8501          ║${NC}"
[ -n "$REACT_PID" ] && \
echo -e "${GREEN}║  React UI   → http://localhost:3000          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Cleanup on exit
cleanup() {
  log "Shutting down services…"
  kill $BACKEND_PID   2>/dev/null || true
  kill $STREAMLIT_PID 2>/dev/null || true
  [ -n "$REACT_PID" ] && kill $REACT_PID 2>/dev/null || true
  log "Done."
}
trap cleanup INT TERM

wait
