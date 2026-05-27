"""
src/models/ml_model.py
Random Forest wait-time prediction model.
  - GridSearchCV hyperparameter tuning
  - Feature importance reporting
  - SHAP explainability (optional)
  - Inference with uncertainty estimate (std across trees)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    FEATURE_COLUMNS, MODELS_DIR, RF_MODEL_PATH, RF_PARAMS,
    RANDOM_STATE, TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


class WaitTimeRFModel:
    """
    Random Forest regressor for EV charging wait-time prediction.
    Uses all trees' predictions to provide a confidence interval.
    """

    def __init__(self, params: dict = None):
        self.params = params or RF_PARAMS
        self.model: Optional[RandomForestRegressor] = None
        self.feature_cols: List[str] = []
        self.feature_importances_: Optional[pd.Series] = None

    # ─── Build ────────────────────────────────────────────────────────────────

    def build(self) -> "WaitTimeRFModel":
        self.model = RandomForestRegressor(**self.params)
        logger.info(f"RF model created with params: {self.params}")
        return self

    # ─── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Optional[List[str]] = None,
    ) -> "WaitTimeRFModel":
        self.feature_cols = feature_cols or [c for c in FEATURE_COLUMNS if c in X_train.columns]
        if not self.feature_cols:
            self.feature_cols = list(X_train.select_dtypes(include=[np.number]).columns)

        X = X_train[self.feature_cols].fillna(0)
        y = y_train.fillna(0)

        if self.model is None:
            self.build()

        logger.info(f"Training RF on {len(X)} samples, {len(self.feature_cols)} features")
        self.model.fit(X, y)

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

        logger.info("RF training complete")
        logger.info(f"Top-5 features:\n{self.feature_importances_.head(5)}")
        return self

    # ─── Hyperparameter Search ────────────────────────────────────────────────

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Optional[List[str]] = None,
        cv: int = 5,
    ) -> "WaitTimeRFModel":
        """Grid search over key hyperparameters."""
        self.feature_cols = feature_cols or [c for c in FEATURE_COLUMNS if c in X_train.columns]
        X = X_train[self.feature_cols].fillna(0)
        y = y_train.fillna(0)

        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [10, 15, None],
            "min_samples_split": [2, 5],
        }
        base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        search = GridSearchCV(base_rf, param_grid, cv=cv, scoring="neg_root_mean_squared_error",
                              n_jobs=-1, verbose=1)
        logger.info("Starting GridSearchCV...")
        search.fit(X, y)
        self.model = search.best_estimator_
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best CV RMSE: {-search.best_score_:.3f}")
        return self

    # ─── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not trained. Call train() first."
        avail = [c for c in self.feature_cols if c in X.columns]
        missing = [c for c in self.feature_cols if c not in X.columns]
        X_pred = X[avail].fillna(0)
        if missing:
            for col in missing:
                X_pred[col] = 0.0
            X_pred = X_pred[self.feature_cols]
        return np.clip(self.model.predict(X_pred), 0, None)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean_prediction, lower_bound, upper_bound)
        using individual tree predictions as a confidence estimate.
        """
        assert self.model is not None
        avail = [c for c in self.feature_cols if c in X.columns]
        X_pred = X[avail].fillna(0)
        tree_preds = np.array([tree.predict(X_pred) for tree in self.model.estimators_])
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        lower = np.clip(mean - 1.96 * std, 0, None)
        upper = mean + 1.96 * std
        return mean, lower, upper

    # ─── Cross-Validate ───────────────────────────────────────────────────────

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        cv: int = 5,
    ) -> Dict[str, float]:
        cols = feature_cols or self.feature_cols
        X_cv = X[cols].fillna(0)
        y_cv = y.fillna(0)
        scores = {}
        for metric in ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]:
            cv_scores = cross_val_score(
                RandomForestRegressor(**self.params), X_cv, y_cv,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            key = metric.replace("neg_", "")
            scores[key] = abs(cv_scores.mean())
        logger.info(f"CV Results: {scores}")
        return scores

    # ─── Explain (SHAP) ───────────────────────────────────────────────────────

    def explain(self, X_sample: pd.DataFrame, n_samples: int = 100) -> Optional[pd.DataFrame]:
        """SHAP-based feature explanation (requires shap package)."""
        try:
            import shap
            avail = [c for c in self.feature_cols if c in X_sample.columns]
            explainer = shap.TreeExplainer(self.model)
            sample = X_sample[avail].fillna(0).head(n_samples)
            shap_values = explainer.shap_values(sample)
            return pd.DataFrame(
                np.abs(shap_values).mean(axis=0),
                index=avail, columns=["mean_shap"],
            ).sort_values("mean_shap", ascending=False)
        except ImportError:
            logger.warning("shap not installed. pip install shap")
            return None

    # ─── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: Path = RF_MODEL_PATH) -> None:
        bundle = {"model": self.model, "feature_cols": self.feature_cols}
        joblib.dump(bundle, path)
        logger.info(f"RF model saved to {path}")

    def load(self, path: Path = RF_MODEL_PATH) -> "WaitTimeRFModel":
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        logger.info(f"RF model loaded from {path}")
        return self


# ─── Gradient Boosting alternative ───────────────────────────────────────────

class WaitTimeGBModel:
    """Gradient Boosting variant (lighter, often comparable accuracy)."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        self.feature_cols: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> "WaitTimeGBModel":
        self.feature_cols = feature_cols
        X_train = X[feature_cols].fillna(0)
        self.model.fit(X_train, y.fillna(0))
        logger.info("GBM training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[self.feature_cols].fillna(0)), 0, None)


if __name__ == "__main__":
    # Smoke test with dummy data
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        "hour_of_day": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "traffic_score": np.random.uniform(0, 1, n),
        "station_utilization": np.random.uniform(0, 1, n),
        "queue_size": np.random.randint(0, 10, n),
        "is_weekend": np.random.randint(0, 2, n),
        "temperature_c": np.random.normal(18, 6, n),
        "hour_sin": np.random.uniform(-1, 1, n),
        "hour_cos": np.random.uniform(-1, 1, n),
    })
    y = pd.Series(X["queue_size"] * 8 + X["traffic_score"] * 15 + np.random.normal(0, 3, n)).clip(0)

    rf = WaitTimeRFModel()
    rf.train(X, y, feature_cols=list(X.columns))
    preds = rf.predict(X.head(5))
    mean, lo, hi = rf.predict_with_uncertainty(X.head(5))
    print("Predictions:", preds)
    print("Uncertainty: lo={}, hi={}".format(lo.round(1), hi.round(1)))
    rf.save()
    print("Feature importances:\n", rf.feature_importances_.head(8))
