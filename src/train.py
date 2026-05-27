"""
src/train.py
Master training script. Runs the full ML pipeline:
  1. Load & preprocess data
  2. Feature engineering
  3. Train Random Forest
  4. Train LSTM
  5. Evaluate both models
  6. Save trained artefacts
  7. Log metrics summary

Usage:
  python src/train.py                        # train both models
  python src/train.py --model rf             # RF only
  python src/train.py --model lstm           # LSTM only
  python src/train.py --tune                 # enable GridSearchCV
  python src/train.py --use-real-data        # skip synthetic generation
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    FEATURE_COLUMNS, LSTM_PARAMS, MODELS_DIR, PROCESSED_FEATURES_CSV,
    RANDOM_STATE, TARGET_COLUMN, TRAIN_CSV, TEST_CSV,
)
from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.ml_model import WaitTimeRFModel
from src.models.lstm_model import LSTMWaitTimeModel, create_sequences, train_val_split_temporal
from src.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EV Wait-Time Prediction Models")
    parser.add_argument("--model", choices=["rf", "lstm", "both"], default="both")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV for RF")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Attempt real API collection (requires keys)")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Load existing processed CSVs directly")
    parser.add_argument("--attention", action="store_true",
                        help="Enable attention in LSTM")
    return parser.parse_args()


def load_or_collect(args) -> pd.DataFrame:
    """Return engineered feature DataFrame."""
    if args.skip_collection and PROCESSED_FEATURES_CSV.exists():
        logger.info(f"Loading existing features: {PROCESSED_FEATURES_CSV}")
        return pd.read_csv(PROCESSED_FEATURES_CSV)

    # 1. Collect
    if args.use_real_data:
        collector = DataCollector()
        collector.collect_all()

    # 2. Preprocess
    preprocessor = DataPreprocessor()
    train_raw, test_raw = preprocessor.run()

    # Combine for feature engineering (split again after FE)
    combined = pd.concat([train_raw, test_raw], ignore_index=True)

    # 3. Feature Engineering
    fe = FeatureEngineer()
    df = fe.run(combined, save=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    avail = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not avail:
        # Fallback: use all numeric columns except target
        avail = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET_COLUMN
        ]
    return avail


def train_rf(df: pd.DataFrame, feature_cols: list, tune: bool = False) -> WaitTimeRFModel:
    logger.info("=" * 60)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 60)

    df_clean = df.dropna(subset=[TARGET_COLUMN])
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rf = WaitTimeRFModel()
    if tune:
        rf.tune(X_train, y_train, feature_cols)
    else:
        rf.build().train(X_train, y_train, feature_cols)

    # Quick test evaluation
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    logger.info(f"RF  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

    rf.save()
    return rf


def train_lstm(df: pd.DataFrame, feature_cols: list, use_attention: bool = False) -> LSTMWaitTimeModel:
    logger.info("=" * 60)
    logger.info("TRAINING LSTM MODEL")
    logger.info("=" * 60)

    #df_clean = df.dropna(subset=[TARGET_COLUMN]).sort_values(
     #   "session_start" if "session_start" in df.columns else df.columns[0]
    #).reset_index(drop=True)
    
    df_clean = df.dropna(subset=[TARGET_COLUMN]).sort_values(
         "start_time" if "start_time" in df.columns else df.columns[0]
    ).reset_index(drop=True)

    seq_len = LSTM_PARAMS["sequence_length"]
    X_seq, y_seq = create_sequences(df_clean, feature_cols, TARGET_COLUMN, seq_len)

    if len(X_seq) < seq_len * 2:
        logger.warning("Not enough data for LSTM sequences — skipping LSTM training")
        return LSTMWaitTimeModel()

    #X_train, y_train, X_val, y_val = train_val_split_temporal(X_seq, y_seq, val_ratio=0.15)
    #X_train_main, X_test, y_train_main, y_test = train_val_split_temporal(
     #   X_train, y_train, val_ratio=0.1
    #)
    # Split sequences properly
    split_idx = int(len(X_seq) * 0.8)

    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]

    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

# Validation split from training data
    val_idx = int(len(X_train) * 0.9)

    X_train_main = X_train[:val_idx]
    y_train_main = y_train[:val_idx]

    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    

    lstm = LSTMWaitTimeModel()
    lstm.build(n_features=X_train.shape[2], use_attention=use_attention)
    lstm.train(X_train_main, y_train_main, X_val, y_val, feature_cols=feature_cols)

    y_pred = lstm.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    logger.info(f"LSTM RMSE={rmse:.3f}  MAE={mae:.3f}")

    lstm.save()
    return lstm


def save_metrics(rf_metrics: dict, lstm_metrics: dict) -> None:
    metrics = {
        "random_forest": rf_metrics,
        "lstm": lstm_metrics,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    path = MODELS_DIR / "training_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


def main():
    args = parse_args()
    t0 = time.time()

    logger.info("🔌 EV ChargeSmart — Model Training Pipeline")
    logger.info(f"Config: model={args.model}, tune={args.tune}, attention={args.attention}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_or_collect(args)
    if df.empty:
        logger.error("Empty dataset — cannot train. Check data collection step.")
        return

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    rf_metrics, lstm_metrics = {}, {}

    # ── RF ────────────────────────────────────────────────────────────────────
    if args.model in ("rf", "both"):
        rf = train_rf(df, feature_cols, tune=args.tune)
        if rf.feature_importances_ is not None:
            rf_metrics["top_features"] = rf.feature_importances_.head(5).to_dict()

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if args.model in ("lstm", "both"):
        lstm = train_lstm(df, feature_cols, use_attention=args.attention)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"✅ Training complete in {elapsed:.1f}s")
    logger.info(f"   Models saved to: {MODELS_DIR}")
    logger.info("=" * 60)

    save_metrics(rf_metrics, lstm_metrics)


if __name__ == "__main__":
    main()
