# models/ — Trained model artefacts
#
# Files generated after running `python src/train.py`:
#   rf_model.pkl          — Trained Random Forest regressor
#   lstm_fallback.pkl     — GBM fallback (when TensorFlow unavailable)
#   lstm_model.h5         — Bidirectional LSTM (if TensorFlow installed)
#   scaler.pkl            — StandardScaler for numeric features
#   label_encoder.pkl     — LabelEncoder for categorical columns
#
# Files generated after running `python src/evaluate.py`:
#   eval_summary.json     — MAE / RMSE / R² for RF and LSTM
#   feature_importance.csv— Ranked feature importances
#   rf_evaluation.png     — Predicted vs actual + residuals + importance
#   heatmap_wait.png      — DoW × Hour average wait heatmap
#   lstm_evaluation.png   — LSTM forecast comparison
#
# Files generated after running `python notebooks/eda.py`:
#   eda_distributions.png — Wait / duration / energy distributions
#   eda_temporal.png      — Hourly / DOW / monthly / weekend patterns
#   eda_stations.png      — Station load vs wait scatter
#   eda_correlation.png   — Feature correlation heatmap
#
# Binary model files (.pkl, .h5) are excluded from git via .gitignore.
# Use DVC or cloud storage (S3 / GCS) for large model versioning.
