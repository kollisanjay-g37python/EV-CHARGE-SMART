# Makefile — EV ChargeSmart project commands
# Usage: make <target>

.PHONY: help install train api streamlit frontend test lint docker clean data

PYTHON   := python
PIP      := pip
UVICORN  := uvicorn
NPM      := npm

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "⚡  EV ChargeSmart — Available Commands"
	@echo "========================================="
	@echo "  make install      Install all Python dependencies"
	@echo "  make data         Collect & preprocess all datasets"
	@echo "  make train        Train RF + LSTM models (full pipeline)"
	@echo "  make train-rf     Train Random Forest only"
	@echo "  make train-lstm   Train LSTM only"
	@echo "  make train-tune   Train RF with GridSearchCV tuning"
	@echo "  make api          Start FastAPI backend (port 8000)"
	@echo "  make streamlit    Start Streamlit dashboard (port 8501)"
	@echo "  make frontend     Start React dev server (port 3000)"
	@echo "  make test         Run all pytest tests"
	@echo "  make test-cov     Run tests with HTML coverage report"
	@echo "  make lint         Run flake8 + black check"
	@echo "  make format       Auto-format code with black + isort"
	@echo "  make docker       Build Docker image"
	@echo "  make docker-up    Start full stack via docker-compose"
	@echo "  make docker-down  Stop docker-compose stack"
	@echo "  make clean        Remove generated files and caches"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r deployment/requirements.txt
	@echo "✅ Dependencies installed"

install-dev: install
	$(PIP) install shap pytest-asyncio httpx
	@echo "✅ Dev dependencies installed"

# ── Environment ───────────────────────────────────────────────────────────────
env:
	@cp -n .env.example .env || true
	@echo "📝 .env created — fill in your API keys"

# ── Data Pipeline ─────────────────────────────────────────────────────────────
data:
	$(PYTHON) -c "from src.data_collection import DataCollector; DataCollector().collect_all()"
	$(PYTHON) -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().run()"
	@echo "✅ Data collected and preprocessed → data/processed/"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) src/train.py --model both
	@echo "✅ RF + LSTM models saved → models/"

train-rf:
	$(PYTHON) src/train.py --model rf
	@echo "✅ Random Forest model saved → models/rf_model.pkl"

train-lstm:
	$(PYTHON) src/train.py --model lstm
	@echo "✅ LSTM model saved → models/lstm_model.h5"

train-tune:
	$(PYTHON) src/train.py --model rf --tune
	@echo "✅ Tuned RF model saved"

train-attention:
	$(PYTHON) src/train.py --model lstm --attention
	@echo "✅ Attention LSTM model saved"

# ── Services ──────────────────────────────────────────────────────────────────
api:
	$(UVICORN) backend.app:app --host 0.0.0.0 --port 8000 --reload
	@echo "🚀 FastAPI running at http://localhost:8000"
	@echo "📚 Docs: http://localhost:8000/docs"

api-prod:
	$(UVICORN) backend.app:app --host 0.0.0.0 --port 8000 --workers 4

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
	@echo "🎨 Streamlit running at http://localhost:8501"

frontend:
	cd frontend && $(NPM) install && $(NPM) start
	@echo "⚛️  React app running at http://localhost:3000"

frontend-build:
	cd frontend && $(NPM) install && $(NPM) run build
	@echo "✅ React build → frontend/build/"

# ── Run all (3 separate terminals needed, or use docker-compose) ──────────────
run-all:
	@echo "⚠️  Use docker-compose for running all services together:"
	@echo "    make docker-up"
	@echo "Or run in separate terminals:"
	@echo "    Terminal 1: make api"
	@echo "    Terminal 2: make streamlit"
	@echo "    Terminal 3: make frontend"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short
	@echo "✅ All tests complete"

test-data:
	pytest tests/test_data_pipeline.py -v --tb=short

test-models:
	pytest tests/test_models.py -v --tb=short

test-api:
	pytest tests/test_api.py -v --tb=short

test-rec:
	pytest tests/test_recommendation.py -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov=backend --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report → htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	flake8 src/ backend/ alerts/ config/ --max-line-length=120 --ignore=E501,W503
	@echo "✅ Lint passed"

format:
	black src/ backend/ alerts/ config/ streamlit_app/ tests/ --line-length=100
	isort src/ backend/ alerts/ config/ streamlit_app/ tests/
	@echo "✅ Code formatted"

typecheck:
	mypy src/ backend/ --ignore-missing-imports --no-strict-optional
	@echo "✅ Type check passed"

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	docker build -f deployment/dockerfile -t ev-chargesmart:latest .
	@echo "✅ Docker image built: ev-chargesmart:latest"

docker-up:
	docker-compose up --build -d
	@echo "🚀 Stack running:"
	@echo "   API       → http://localhost:8000"
	@echo "   Docs      → http://localhost:8000/docs"
	@echo "   Streamlit → http://localhost:8501"
	@echo "   Redis     → localhost:6379"

docker-up-dev:
	docker-compose --profile dev up --build -d
	@echo "🚀 Dev stack with React → http://localhost:3000"

docker-down:
	docker-compose down
	@echo "🛑 Stack stopped"

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker rmi ev-chargesmart:latest || true
	@echo "🗑  Docker resources cleaned"

# ── Notebooks ─────────────────────────────────────────────────────────────────
notebooks:
	jupyter lab notebooks/ --port 8888
	@echo "📓 JupyterLab at http://localhost:8888"

# ── Utilities ─────────────────────────────────────────────────────────────────
predict-demo:
	$(PYTHON) -c "\
from src.predict import PredictionEngine; \
e = PredictionEngine(); e.load_models(); \
r = e.predict_single(1, 8, 3, 4, 18, 2, 0.72); \
print('Demo prediction:', r)"

queue-demo:
	$(PYTHON) -c "\
from src.queue_model import MMcQueueModel; \
m = MMcQueueModel(); \
s = m.compute_wait(4, 8.0, 3.5, current_queue=3, station_id=1, station_name='Test'); \
print('Queue state:'); \
[print(f'  {k}: {v}') for k,v in s.to_dict().items()]"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -f data/real_time_cache/*.json
	@echo "🗑  Cleaned cache and compiled files"

clean-models:
	rm -f models/rf_model.pkl models/lstm_model.h5 models/scaler.pkl models/label_encoder.pkl
	@echo "🗑  Model files removed — re-run make train"

clean-data:
	rm -rf data/processed/ data/real_time_cache/
	@echo "🗑  Processed data cleaned — re-run make data"

clean-all: clean clean-models clean-data
	@echo "🗑  Full clean complete"
