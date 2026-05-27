// frontend/src/components/StationDetail.js
// Full page detail view for a single charging station.
// Shows live prediction, queue analysis, 12-step forecast, and navigation.

import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useApp } from "../App";
import { fetchPrediction, fetchForecast, fetchQueueAnalysis } from "../services/api";
import WaitTimeBadge from "./WaitTimeBadge";
import ForecastChart from "./ForecastChart";
import QueueGauge from "./QueueGauge";
import MetricCard from "./MetricCard";

export default function StationDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { stations, API_BASE } = useApp();

  const station = stations.find((s) => String(s.station_id) === String(id));
  const [prediction, setPrediction] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [queueData, setQueueData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    if (!station) return;
    const load = async () => {
      setLoading(true);
      try {
        const [pred, fore, queue] = await Promise.all([
          fetchPrediction(API_BASE, {
            station_id: station.station_id,
            num_ports: station.num_ports,
            available_ports: station.available_ports,
            queue_size: station.queue_size || 0,
            hour: new Date().getHours(),
            day_of_week: new Date().getDay(),
            traffic_score: parseFloat(station.traffic_score) || 0.5,
          }),
          fetchForecast(API_BASE, station.station_id),
          fetchQueueAnalysis(API_BASE, station.station_id, 8.0, 3.0),
        ]);
        setPrediction(pred);
        setForecast(fore.forecast || []);
        setQueueData(queue);
      } catch (e) {
        console.warn("API error:", e);
        setPrediction(mockPrediction(station));
        setForecast(mockForecast());
        setQueueData(mockQueue());
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [station, API_BASE]);

  if (!station) {
    return (
      <div className="not-found">
        <h2>Station not found</h2>
        <button className="btn-primary" onClick={() => navigate("/")}>← Back</button>
      </div>
    );
  }

  const utilPct = station.num_ports
    ? (((station.num_ports - station.available_ports) / station.num_ports) * 100).toFixed(0)
    : 0;
  const statusColor = station.status === "Operational" ? "#22c55e"
    : station.status === "Partial" ? "#f59e0b" : "#ef4444";

  return (
    <div className="station-detail">
      {/* Header */}
      <div className="detail-header">
        <button className="back-btn" onClick={() => navigate(-1)}>← Back</button>
        <div className="detail-title-row">
          <div>
            <h1 className="detail-name">{station.name}</h1>
            <div className="detail-meta">
              <span className="connector-badge">{station.connector_type}</span>
              <span className="power-badge">⚡ {station.power_kw} kW</span>
              <span className="operator-badge">🏢 {station.operator}</span>
              <span className="status-badge" style={{ color: statusColor }}>
                ● {station.status}
              </span>
            </div>
          </div>
          <a
            className="btn-primary nav-btn"
            href={`https://www.google.com/maps/dir/?api=1&destination=${station.lat},${station.lng}`}
            target="_blank" rel="noreferrer"
          >
            🗺 Navigate
          </a>
        </div>
      </div>

      {/* KPI Row */}
      <div className="detail-kpi-grid">
        <MetricCard
          icon="⏱" label="Predicted Wait"
          value={loading ? "…" : `${prediction?.predicted_wait_min ?? "—"} min`}
          color="#00d4aa"
        />
        <MetricCard
          icon="🟢" label="Available Ports"
          value={`${station.available_ports}/${station.num_ports}`}
          sub={`${utilPct}% utilized`} color="#22c55e"
        />
        <MetricCard
          icon="🚗" label="Queue Size"
          value={station.queue_size || 0}
          sub="vehicles waiting" color="#f59e0b"
        />
        <MetricCard
          icon="📶" label="Traffic Score"
          value={`${(parseFloat(station.traffic_score) * 100).toFixed(0)}%`}
          sub="congestion level" color="#3b82f6"
        />
        <MetricCard
          icon="🎯" label="Recommendation"
          value={loading ? "…" : (prediction?.recommendation || "—")}
          color={recColor(prediction?.recommendation)}
        />
        <MetricCard
          icon="📍" label="Coordinates"
          value={`${station.lat?.toFixed(4)}, ${station.lng?.toFixed(4)}`}
          color="#8b5cf6"
        />
      </div>

      {/* Tabs */}
      <div className="tab-bar">
        {[
          { key: "overview",  label: "📊 Overview"  },
          { key: "forecast",  label: "📈 Forecast"  },
          { key: "queue",     label: "🚗 Queue"     },
          { key: "raw",       label: "🔧 Raw Data"  },
        ].map(({ key, label }) => (
          <button
            key={key}
            className={`tab-btn ${activeTab === key ? "active" : ""}`}
            onClick={() => setActiveTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="tab-content">

        {/* Overview */}
        {activeTab === "overview" && (
          <div className="detail-overview-grid">
            <div>
              <h3 className="section-label">Utilization Bar</h3>
              <div className="util-bar-wrap">
                <div className="util-bar-bg">
                  <div
                    className="util-bar-fill"
                    style={{
                      width: `${utilPct}%`,
                      background: utilPct > 85 ? "#ef4444" : utilPct > 60 ? "#f59e0b" : "#22c55e",
                    }}
                  />
                </div>
                <span className="util-label">{utilPct}%</span>
              </div>

              <h3 className="section-label" style={{ marginTop: 24 }}>Prediction Breakdown</h3>
              {prediction && (
                <div className="pred-breakdown">
                  {[
                    { label: "🌲 Random Forest", value: `${prediction.rf_prediction ?? "N/A"} min`, weight: "40%" },
                    { label: "🧠 LSTM Model",    value: "—",                                          weight: "45%" },
                    { label: "📐 M/M/c Queue",   value: `${prediction.queue_prediction?.toFixed(1)} min`, weight: "15%" },
                  ].map(({ label, value, weight }) => (
                    <div key={label} className="pred-row">
                      <span className="pred-label">{label}</span>
                      <span className="pred-value">{value}</span>
                      <span className="pred-weight">{weight} weight</span>
                    </div>
                  ))}
                  <div className="pred-ensemble">
                    <span>⚡ Ensemble Result</span>
                    <strong>{prediction.predicted_wait_min} min</strong>
                  </div>
                </div>
              )}
            </div>

            <div>
              <h3 className="section-label">Station Info</h3>
              <div className="info-table">
                {[
                  ["Station ID",      station.station_id],
                  ["Operator",        station.operator || "—"],
                  ["Connector",       station.connector_type],
                  ["Power Level",     `${station.power_kw} kW`],
                  ["Total Ports",     station.num_ports],
                  ["Available",       station.available_ports],
                  ["Status",          station.status],
                  ["Traffic Score",   parseFloat(station.traffic_score).toFixed(3)],
                  ["Latitude",        station.lat?.toFixed(5)],
                  ["Longitude",       station.lng?.toFixed(5)],
                ].map(([k, v]) => (
                  <div key={k} className="info-row">
                    <span className="info-key">{k}</span>
                    <span className="info-val">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Forecast */}
        {activeTab === "forecast" && (
          <div>
            <h3 className="section-label">12-Hour Wait Time Forecast</h3>
            <ForecastChart data={forecast} currentHour={new Date().getHours()} expanded />
            <div className="forecast-grid">
              {forecast.map((f) => (
                <div
                  key={f.hour}
                  className={`forecast-cell ${f.hour === new Date().getHours() ? "current" : ""}`}
                >
                  <div className="fc-label">{f.label || `${f.hour}:00`}</div>
                  <WaitTimeBadge wait={parseFloat(f.predicted_wait_min)} />
                  <div className="fc-rec">{f.recommendation}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Queue */}
        {activeTab === "queue" && queueData && (
          <div>
            <h3 className="section-label">M/M/c Queue Analysis (Erlang-C)</h3>
            <QueueGauge data={queueData} expanded />
            <div className="queue-formula-box">
              <code>
                ρ = λ / (c·μ) = {queueData.arrival_rate?.toFixed(1)} /
                ({queueData.num_ports} × {queueData.service_rate?.toFixed(1)}) =
                {queueData.rho?.toFixed(4)}
              </code>
              <br />
              <code>Wq = C(c,ρ) / (c·μ − λ) = {queueData.avg_wait_min?.toFixed(2)} min</code>
            </div>
          </div>
        )}

        {/* Raw Data */}
        {activeTab === "raw" && (
          <div>
            <h3 className="section-label">Raw Station Data (JSON)</h3>
            <pre className="json-block">{JSON.stringify(station, null, 2)}</pre>
            {prediction && (
              <>
                <h3 className="section-label">Prediction Response (JSON)</h3>
                <pre className="json-block">{JSON.stringify(prediction, null, 2)}</pre>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const recColor = (rec) => ({
  GO_NOW: "#22c55e", GOOD_TIME: "#3b82f6",
  MODERATE_WAIT: "#f59e0b", LONG_WAIT: "#f97316", AVOID: "#ef4444",
}[rec] || "#64748b");

const mockPrediction = (station) => ({
  predicted_wait_min: Math.floor(Math.random() * 30 + 5),
  rf_prediction: Math.floor(Math.random() * 30 + 5),
  queue_prediction: Math.floor(Math.random() * 20 + 3),
  recommendation: "MODERATE_WAIT",
  utilization_pct: 72.0,
  queue_stable: true,
});

const mockForecast = () =>
  Array.from({ length: 12 }, (_, i) => {
    const h = (new Date().getHours() + i) % 24;
    const w = Math.max(2, 20 + 10 * Math.sin(h / 4) + Math.random() * 5);
    return { hour: h, label: `${h}:00`, predicted_wait_min: w.toFixed(1), recommendation: w < 15 ? "GO_NOW" : "MODERATE_WAIT" };
  });

const mockQueue = () => ({
  rho: 0.68, erlang_c: 0.38, avg_wait_min: 10.2,
  avg_queue_length: 1.4, throughput_per_hr: 7.5,
  system_stable: true, num_ports: 6,
  arrival_rate: 8.0, service_rate: 3.5,
});
