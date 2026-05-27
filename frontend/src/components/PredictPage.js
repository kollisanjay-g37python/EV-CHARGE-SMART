// frontend/src/components/PredictPage.js
// Interactive wait-time prediction form.
// Calls POST /api/v1/predict and shows RF + LSTM + Queue ensemble result.

import React, { useState } from "react";
import { useApp } from "../App";
import { fetchPrediction } from "../services/api";
import WaitTimeBadge from "./WaitTimeBadge";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const CONNECTORS = ["Type 2", "CCS", "CHAdeMO", "Tesla", "J1772"];

const REC_STYLES = {
  GO_NOW:        { color: "#22c55e", icon: "⚡" },
  GOOD_TIME:     { color: "#00d4aa", icon: "✅" },
  MODERATE_WAIT: { color: "#f59e0b", icon: "⏳" },
  LONG_WAIT:     { color: "#ef4444", icon: "🔴" },
  AVOID:         { color: "#ef4444", icon: "🚫" },
};

export default function PredictPage() {
  const { stations, API_BASE } = useApp();

  const [form, setForm] = useState({
    station_id: 1,
    num_ports: 8,
    available_ports: 3,
    queue_size: 2,
    hour: new Date().getHours(),
    day_of_week: new Date().getDay() === 0 ? 6 : new Date().getDay() - 1,
    traffic_score: 0.5,
    temperature_c: 20,
    precipitation_mm: 0,
    connector_type: "CCS",
    lat: 37.7749,
    lng: -122.4194,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }));

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchPrediction(API_BASE, form);
      setResult(data);
    } catch (err) {
      setError("Prediction failed — check API connection.");
    } finally {
      setLoading(false);
    }
  };

  const recStyle = result ? (REC_STYLES[result.recommendation] || REC_STYLES.MODERATE_WAIT) : null;

  return (
    <div className="predict-page">
      <div className="page-header">
        <h1 className="page-title">⚡ Live Wait-Time Predictor</h1>
        <p className="page-subtitle">
          Ensemble: Random Forest (45%) + LSTM (40%) + M/M/c Queue (15%)
        </p>
      </div>

      <div className="predict-layout">
        {/* ── Form ─────────────────────────────────────────────────────────── */}
        <div className="predict-form card">
          <h2>Input Features</h2>

          {/* Station selector */}
          <div className="form-group">
            <label>Station</label>
            <select value={form.station_id} onChange={(e) => {
              const sid = Number(e.target.value);
              const st = stations.find((s) => s.station_id === sid);
              update("station_id", sid);
              if (st) {
                update("num_ports", st.num_ports || 8);
                update("available_ports", st.available_ports ?? 3);
                update("queue_size", st.queue_size ?? 2);
                update("connector_type", st.connector_type || "CCS");
                update("lat", st.lat || 37.7749);
                update("lng", st.lng || -122.4194);
              }
            }}>
              {stations.map((s) => (
                <option key={s.station_id} value={s.station_id}>{s.name}</option>
              ))}
            </select>
          </div>

          {/* Port inputs */}
          <div className="form-row">
            <div className="form-group">
              <label>Total Ports</label>
              <input type="number" min={1} max={50} value={form.num_ports}
                onChange={(e) => update("num_ports", Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Available Ports</label>
              <input type="number" min={0} max={form.num_ports} value={form.available_ports}
                onChange={(e) => update("available_ports", Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Queue Size</label>
              <input type="number" min={0} max={50} value={form.queue_size}
                onChange={(e) => update("queue_size", Number(e.target.value))} />
            </div>
          </div>

          {/* Time inputs */}
          <div className="form-row">
            <div className="form-group">
              <label>Hour of Day: {form.hour}:00</label>
              <input type="range" min={0} max={23} value={form.hour}
                onChange={(e) => update("hour", Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Day of Week</label>
              <select value={form.day_of_week}
                onChange={(e) => update("day_of_week", Number(e.target.value))}>
                {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
              </select>
            </div>
          </div>

          {/* Traffic & weather */}
          <div className="form-group">
            <label>Traffic Score: {(form.traffic_score * 100).toFixed(0)}%</label>
            <div className="slider-row">
              <span className="slider-min">🟢 Free Flow</span>
              <input type="range" min={0} max={1} step={0.01} value={form.traffic_score}
                onChange={(e) => update("traffic_score", Number(e.target.value))} />
              <span className="slider-max">🔴 Congested</span>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Temperature (°C): {form.temperature_c}</label>
              <input type="range" min={-10} max={45} value={form.temperature_c}
                onChange={(e) => update("temperature_c", Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Precipitation (mm): {form.precipitation_mm}</label>
              <input type="range" min={0} max={20} value={form.precipitation_mm}
                onChange={(e) => update("precipitation_mm", Number(e.target.value))} />
            </div>
          </div>

          <div className="form-group">
            <label>Connector Type</label>
            <select value={form.connector_type}
              onChange={(e) => update("connector_type", e.target.value)}>
              {CONNECTORS.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <button
            className="btn-predict"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "⚙️ Running Models..." : "⚡ Predict Wait Time"}
          </button>
          {error && <div className="form-error">{error}</div>}
        </div>

        {/* ── Result ───────────────────────────────────────────────────────── */}
        <div className="predict-result card">
          <h2>Prediction Output</h2>

          {!result && !loading && (
            <div className="result-empty">
              <div className="result-empty-icon">⚡</div>
              <p>Configure inputs and click Predict</p>
            </div>
          )}

          {loading && (
            <div className="result-loading">
              <div className="spinner-icon">⚙️</div>
              <p>Running RF + LSTM + Queue ensemble...</p>
            </div>
          )}

          {result && (
            <>
              {/* Big result */}
              <div className="result-hero" style={{ borderColor: recStyle.color }}>
                <div className="result-hero-label">Ensemble Prediction</div>
                <div className="result-hero-value" style={{ color: recStyle.color }}>
                  {result.predicted_wait_min?.toFixed(1)}
                </div>
                <div className="result-hero-unit">minutes wait</div>
                <div className="result-hero-rec" style={{ color: recStyle.color }}>
                  {recStyle.icon} {result.recommendation?.replace(/_/g, " ")}
                </div>
              </div>

              {/* Model breakdown */}
              <div className="result-breakdown">
                {[
                  { label: "🌲 Random Forest", value: result.rf_prediction, weight: "45%", color: "#22c55e" },
                  { label: "📐 M/M/c Queue",   value: result.queue_prediction?.toFixed(1), weight: "15%", color: "#3b82f6" },
                ].map(({ label, value, weight, color }) => (
                  <div className="breakdown-card" key={label}>
                    <div className="bc-label">{label}</div>
                    <div className="bc-value" style={{ color }}>{value ?? "N/A"} min</div>
                    <div className="bc-weight">Weight: {weight}</div>
                  </div>
                ))}
              </div>

              {/* Metadata */}
              <div className="result-meta">
                <div className="meta-row"><span>Utilisation</span><span>{result.utilization_pct?.toFixed(1)}%</span></div>
                <div className="meta-row"><span>Queue Stable</span><span>{result.queue_stable ? "Yes ✅" : "No ⚠️"}</span></div>
                <div className="meta-row"><span>Confidence</span><span>{result.confidence_level}</span></div>
                <div className="meta-row"><span>Timestamp</span><span>{new Date(result.timestamp).toLocaleTimeString()}</span></div>
              </div>

              {/* JSON response */}
              <details className="result-json">
                <summary>📋 Raw API Response (JSON)</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
