"""
src/models/lstm_model.py
LSTM neural network for time-series wait-time prediction.
Architecture:
  Input  → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)
Supports:
  - Multi-step ahead forecasting
  - Attention mechanism (optional)
  - Early stopping + LR scheduling
  - Quantile prediction for uncertainty bounds
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    FEATURE_COLUMNS, LSTM_MODEL_PATH, LSTM_PARAMS,
    MODELS_DIR, RANDOM_STATE, TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── TensorFlow import guard ─────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import (
        LSTM, Attention, Dense, Dropout, Input,
        Bidirectional, LayerNormalization, MultiHeadAttention,
    )
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    tf.random.set_seed(RANDOM_STATE)
    logger.info(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. pip install tensorflow")


class LSTMWaitTimeModel:
    """
    Bidirectional LSTM for sequence-to-one wait-time prediction.
    Falls back to a numpy-based mock when TensorFlow is unavailable.
    """

    def __init__(self, params: dict = None):
        self.p = params or LSTM_PARAMS
        self.model = None
        self.feature_cols: List[str] = []
        self.history = None
        self._is_mock = not TF_AVAILABLE

    # ─── Architecture ─────────────────────────────────────────────────────────

    def build(self, n_features: int, use_attention: bool = False) -> "LSTMWaitTimeModel":
        if not TF_AVAILABLE:
            logger.warning("TF unavailable — using mock LSTM")
            return self

        seq_len = self.p["sequence_length"]
        inputs = Input(shape=(seq_len, n_features), name="sequence_input")

        # Bidirectional LSTM block 1
        x = Bidirectional(
            LSTM(self.p["lstm_units_1"], return_sequences=True, name="lstm_1"),
            name="bilstm_1",
        )(inputs)
        x = LayerNormalization()(x)
        x = Dropout(self.p["dropout_rate"])(x)

        # Optional multi-head attention
        if use_attention:
            attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = x + attn_out
            x = LayerNormalization()(x)

        # LSTM block 2
        x = Bidirectional(
            LSTM(self.p["lstm_units_2"], return_sequences=False, name="lstm_2"),
            name="bilstm_2",
        )(x)
        x = Dropout(self.p["dropout_rate"])(x)

        # Dense head
        x = Dense(self.p["dense_units"], activation="relu")(x)
        x = Dense(16, activation="relu")(x)
        outputs = Dense(1, activation="linear", name="wait_time")(x)

        self.model = Model(inputs, outputs, name="ev_lstm")
        self.model.compile(
            optimizer=Adam(learning_rate=self.p["learning_rate"]),
            loss="huber",            # robust to outliers
            metrics=["mae", "mse"],
        )
        logger.info(self.model.summary())
        logger.info(f"Model built: seq_len={seq_len}, n_features={n_features}")
        return self

    # ─── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> "LSTMWaitTimeModel":
        self.feature_cols = feature_cols or []

        if self._is_mock:
            logger.warning("Mock training — TF not available")
            return self

        if self.model is None:
            n_features = X_train.shape[2]
            self.build(n_features)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.p["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=1,
            ),
            ModelCheckpoint(
                str(LSTM_MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info(
            f"Training LSTM: {X_train.shape[0]} train / {X_val.shape[0]} val samples"
        )
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.p["epochs"],
            batch_size=self.p["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("LSTM training complete")
        return self

    # ─── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._is_mock:
            return self._mock_predict(X)
        assert self.model is not None
        raw = self.model.predict(X, verbose=0).flatten()
        return np.clip(raw, 0, None)

    def predict_multistep(
        self, X_seed: np.ndarray, steps: int = 12
    ) -> np.ndarray:
        """
        Autoregressive multi-step forecast.
        Appends each prediction as the new last step input.
        """
        if self._is_mock:
            return np.random.uniform(5, 40, steps)

        assert self.model is not None
        seq = X_seed.copy()   # (1, seq_len, n_features)
        preds = []
        for _ in range(steps):
            y_pred = self.model.predict(seq, verbose=0)[0, 0]
            preds.append(max(0, y_pred))
            # Roll sequence forward: drop oldest step, append new
            new_step = seq[:, -1:, :].copy()
            new_step[0, 0, -1] = y_pred   # overwrite target position
            seq = np.concatenate([seq[:, 1:, :], new_step], axis=1)
        return np.array(preds)

    def _mock_predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if X.ndim >= 1 else 1
        return np.random.uniform(5, 35, n)

    # ─── Training Curves ─────────────────────────────────────────────────────

    def get_training_history(self) -> Optional[pd.DataFrame]:
        if self.history is None:
            return None
        return pd.DataFrame(self.history.history)

    # ─── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: Path = LSTM_MODEL_PATH) -> None:
        if self._is_mock or self.model is None:
            logger.warning("No TF model to save")
            return
        self.model.save(str(path))
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: Path = LSTM_MODEL_PATH) -> "LSTMWaitTimeModel":
        if not TF_AVAILABLE:
            logger.warning("TF unavailable — cannot load LSTM")
            self._is_mock = True
            return self
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"LSTM model not found at {path}")
        self.model = load_model(str(p))
        self._is_mock = False
        logger.info(f"LSTM model loaded from {path}")
        return self


# ─── Data Prep Utilities ──────────────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COLUMN,
    seq_len: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from a time-ordered DataFrame."""
    seq_len = seq_len or LSTM_PARAMS["sequence_length"]
    avail = [c for c in feature_cols if c in df.columns]
    X_data = df[avail].fillna(0).values
    y_data = df[target_col].fillna(0).values

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_data)):
        X_seq.append(X_data[i - seq_len: i])
        y_seq.append(y_data[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def train_val_split_temporal(
    X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-respecting split (no shuffling for sequences)."""
    split = int(len(X) * (1 - val_ratio))
    return X[:split], y[:split], X[split:], y[split:]


if __name__ == "__main__":
    # Smoke test
    seq_len = LSTM_PARAMS["sequence_length"]
    n_feats = 12
    n_samples = 500

    X_dummy = np.random.rand(n_samples, seq_len, n_feats).astype(np.float32)
    y_dummy = np.random.uniform(0, 45, n_samples).astype(np.float32)

    X_train, y_train, X_val, y_val = train_val_split_temporal(X_dummy, y_dummy)

    lstm = LSTMWaitTimeModel()
    if TF_AVAILABLE:
        lstm.build(n_features=n_feats, use_attention=True)
        lstm.train(X_train, y_train, X_val, y_val)
        preds = lstm.predict(X_val[:5])
        print("LSTM predictions:", preds)
        multistep = lstm.predict_multistep(X_val[:1], steps=6)
        print("Multi-step forecast:", multistep)
    else:
        preds = lstm.predict(X_dummy[:5])
        print("Mock predictions:", preds)
