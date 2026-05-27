"""
src/evaluate.py
Comprehensive model evaluation:
  - Regression metrics (RMSE, MAE, MAPE, R²)
  - Residual analysis
  - Segment-level accuracy (by hour / station type)
  - Comparison table: RF vs LSTM vs Queue Model
  - Matplotlib visualisation helpers
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


class ModelEvaluator:
    """Evaluate and compare EV wait-time prediction models."""

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, float]:
        """Return full regression metric dictionary."""
        y_true = np.array(y_true, dtype=float)
        y_pred = np.clip(np.array(y_pred, dtype=float), 0, None)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        mape = _mape(y_true, y_pred)
        smape = _smape(y_true, y_pred)
        within_5 = float(np.mean(np.abs(y_true - y_pred) <= 5) * 100)
        within_10 = float(np.mean(np.abs(y_true - y_pred) <= 10) * 100)

        metrics = {
            "model": model_name,
            "n_samples": len(y_true),
            "rmse_min": round(rmse, 3),
            "mae_min": round(mae, 3),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 2),
            "smape_pct": round(smape, 2),
            "within_5min_pct": round(within_5, 1),
            "within_10min_pct": round(within_10, 1),
        }
        logger.info(f"[{model_name}] RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  "
                    f"MAPE={mape:.1f}%  ±5min={within_5:.1f}%")
        return metrics

    def segment_analysis(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        segment_col: str = "hour_of_day",
        target_col: str = "wait_time_minutes",
    ) -> pd.DataFrame:
        """Compute RMSE per segment (e.g., per hour of day)."""
        df = df.copy()
        df["_pred"] = y_pred
        results = []
        for seg_val, group in df.groupby(segment_col):
            if target_col not in group.columns:
                continue
            y_t = group[target_col].values
            y_p = group["_pred"].values
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            mae = float(mean_absolute_error(y_t, y_p))
            results.append({
                segment_col: seg_val,
                "n": len(group),
                "rmse": round(rmse, 3),
                "mae": round(mae, 3),
            })
        return pd.DataFrame(results).sort_values(segment_col)

    def compare_models(self, comparisons: Dict[str, Dict]) -> pd.DataFrame:
        """
        Build comparison table.
        comparisons = {"RF": metrics_dict, "LSTM": metrics_dict, ...}
        """
        rows = []
        for name, m in comparisons.items():
            rows.append({
                "Model": name,
                "RMSE (min)": m.get("rmse_min"),
                "MAE (min)": m.get("mae_min"),
                "R²": m.get("r2"),
                "MAPE (%)": m.get("mape_pct"),
                "±5 min (%)": m.get("within_5min_pct"),
                "±10 min (%)": m.get("within_10min_pct"),
            })
        df = pd.DataFrame(rows).set_index("Model")
        logger.info("\n" + df.to_string())
        return df

    def residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Return residual statistics for diagnostic plots."""
        residuals = y_true - y_pred
        return pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residuals,
            "abs_residual": np.abs(residuals),
            "pct_error": np.where(y_true != 0, residuals / y_true * 100, np.nan),
        })

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Generate actual vs predicted scatter and residual plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            residuals = y_true - y_pred
            fig = plt.figure(figsize=(14, 5))
            gs = gridspec.GridSpec(1, 3, figure=fig)

            # Scatter: actual vs predicted
            ax1 = fig.add_subplot(gs[0])
            ax1.scatter(y_true, y_pred, alpha=0.3, s=8, color="#00d4aa")
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax1.plot(lims, lims, "r--", lw=1, label="Perfect")
            ax1.set_xlabel("Actual Wait (min)")
            ax1.set_ylabel("Predicted Wait (min)")
            ax1.set_title(f"{model_name}: Actual vs Predicted")
            ax1.legend()

            # Residual histogram
            ax2 = fig.add_subplot(gs[1])
            ax2.hist(residuals, bins=40, color="#3b82f6", alpha=0.8, edgecolor="white")
            ax2.axvline(0, color="red", lw=1.5, linestyle="--")
            ax2.set_xlabel("Residual (min)")
            ax2.set_ylabel("Count")
            ax2.set_title("Residual Distribution")

            # Residuals vs predicted
            ax3 = fig.add_subplot(gs[2])
            ax3.scatter(y_pred, residuals, alpha=0.3, s=8, color="#f59e0b")
            ax3.axhline(0, color="red", lw=1.5, linestyle="--")
            ax3.set_xlabel("Predicted Wait (min)")
            ax3.set_ylabel("Residual (min)")
            ax3.set_title("Residuals vs Predicted")

            plt.suptitle(f"{model_name} Evaluation", fontsize=13, fontweight="bold")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib not installed — skipping plots")

    def plot_feature_importance(
        self,
        importances: pd.Series,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            top = importances.head(top_n)
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#00d4aa" if i == 0 else "#3b82f6" if i < 5 else "#64748b"
                      for i in range(len(top))]
            bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top Feature Importances (Random Forest)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib not installed")

    def plot_lstm_history(
        self,
        history_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history_df["loss"], label="Train Loss", color="#00d4aa")
            if "val_loss" in history_df.columns:
                axes[0].plot(history_df["val_loss"], label="Val Loss", color="#ef4444")
            axes[0].set_title("Training Loss (Huber)")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            if "mae" in history_df.columns:
                axes[1].plot(history_df["mae"], label="Train MAE", color="#3b82f6")
            if "val_mae" in history_df.columns:
                axes[1].plot(history_df["val_mae"], label="Val MAE", color="#f59e0b")
            axes[1].set_title("MAE (minutes)")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.suptitle("LSTM Training Curves", fontsize=13, fontweight="bold")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib not installed")


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.random.uniform(0, 45, n)
    y_pred_rf = y_true + np.random.normal(0, 4, n)
    y_pred_lstm = y_true + np.random.normal(0, 3.5, n)

    ev = ModelEvaluator()
    rf_m = ev.compute_metrics(y_true, y_pred_rf, "RF")
    lstm_m = ev.compute_metrics(y_true, y_pred_lstm, "LSTM")
    print(ev.compare_models({"RF": rf_m, "LSTM": lstm_m}).to_string())
