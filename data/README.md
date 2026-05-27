# data/

Data storage for the EV Charging Smart System.

## Structure

```
data/
├── raw/                    ← Raw data from APIs & synthetic generator
│   ├── stations.csv        — Station locations & metadata (Open Charge Map)
│   └── charging_sessions.csv — Session history (real or synthetic)
│
├── processed/              ← Cleaned & feature-engineered data
│   ├── sessions_clean.csv  — After data_preprocessing.py
│   └── features.csv        — After feature_engineering.py (model input)
│
└── real_time_cache/        ← 5-minute TTL API cache
    ├── traffic_*.json      — TomTom traffic index per location
    └── weather_*.json      — OpenWeatherMap conditions per location
```

## Generating Data

```bash
# Option 1: Full pipeline (recommended)
python src/train.py

# Option 2: Data only
python src/data_collection.py      # → raw/
python src/data_preprocessing.py   # → processed/sessions_clean.csv
python src/feature_engineering.py  # → processed/features.csv
```

## Real Datasets (optional drop-in replacements)

| File | Source | Description |
|---|---|---|
| `raw/stations.csv` | [Open Charge Map](https://openchargemap.org/site/develop/api) | Real station metadata |
| `raw/charging_sessions.csv` | [Kaggle EV Load](https://www.kaggle.com/datasets/datasetengineer/ev-charging-load-dataset-and-optimal-routing) | Real session records |

The system automatically falls back to synthetic data when API keys are missing.
Column schemas must match those produced by `data_collection.py`.
