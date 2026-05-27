// frontend/src/components/Dashboard.js
// Main dashboard — network KPIs, hourly demand chart,
// station status grid, and quick-predict panel.

import React, { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../App";
import {
  fetchStations, fetchForecast, fetchQueueAnalysis,
} from "../services/api";
import StationCard from "./StationCard";
import WaitTimeBadge from "./WaitTimeBadge";
import QueueGauge from "./QueueGauge";
import ForecastChart from "./ForecastChart";
import RecommendationList from "./RecommendationList";

export default function Dashboard() {
  const { stations, userLocation, loading, apiOnline, API_BASE } = useApp();
  const navigate = useNavigate();

  const [forecast, setForecast] = useState([]);
  const [selectedForecastStation, setSelectedForecastStation] = useState(1);
  const [networkStats, setNetworkStats] = useState(null);
  const [sortBy, setSortBy] = useState("wait");
  const [refreshTs, setRefreshTs] = useState(Date.now());

  // ─── Network stats ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!stations.length) return;
    const totalPorts = stations.reduce((a, s) => a + (s.num_ports || 0), 0);
    const totalAvail = stations.reduce((a, s) => a + (s.available_ports || 0), 0);
    const totalQueue = stations.reduce((a, s) => a + (s.queue_size || 0), 0);
    const avgWait = stations.reduce((a, s) => a + (s.wait_time_minutes || s.predicted_wait_min || 0), 0) / stations.length;
    const operational = stations.filter((s) => s.status?.includes("Operational")).length;
    setNetworkStats({ totalPorts, totalAvail, totalQueue, avgWait, operational });
  }, [stations]);

  // ─── Load forecast ──────────────────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchForecast(API_BASE, selectedForecastStation);
        setForecast(data.forecast || []);
      } catch {
        setForecast(generateMockForecast());
      }
    };
    load();
  }, [selectedForecastStation, API_BASE, refreshTs]);

  // ─── Sort stations ──────────────────────────────────────────────────────────
  const sortedStations = [...stations].sort((a, b) => {
    if (sortBy === "wait") return (a.wait_time_minutes || 0) - (b.wait_time_minutes || 0);
    if (sortBy === "distance") return (a.distance_km || 0) - (b.distance_km || 0);
    if (sortBy === "availability") return (b.available_ports || 0) - (a.available_ports || 0);
    return 0;
  });

  // ─── Manual refresh ────────────────────────────────────────────────────────
  const refresh = useCallback(() => setRefreshTs(Date.now()), []);

  if (loading) return <LoadingScreen />;

  return (
    <div className="dashboard">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="dashboard-header">
        <div>
          <h1 className="page-title">⚡ Network Dashboard</h1>
          <p className="page-subtitle">
            ML-powered wait-time predictions · RF + LSTM + M/M/c Queue
            {!apiOnline && <span className="badge-offline"> (Demo Mode)</span>}
          </p>
        </div>
        <button className="btn-icon" onClick={refresh} title="Refresh">
          🔄
        </button>
      </div>

      {/* ── KPI Cards ──────────────────────────────────────────────────────── */}
      {networkStats && (
        <div className="kpi-grid">
          {[
            { icon: "🏢", label: "Stations", value: stations.length, sub: `${networkStats.operational} operational`, color: "#00d4aa" },
            { icon: "🟢", label: "Available Ports", value: `${networkStats.totalAvail}/${networkStats.totalPorts}`, sub: `${((networkStats.totalAvail / Math.max(networkStats.totalPorts, 1)) * 100).toFixed(0)}% free`, color: "#22c55e" },
            { icon: "🚗", label: "Vehicles Waiting", value: networkStats.totalQueue, sub: "across network", color: "#f59e0b" },
            { icon: "⏱", label: "Avg Wait Time", value: `${networkStats.avgWait.toFixed(1)}m`, sub: "ML predicted", color: "#3b82f6" },
            { icon: "📊", label: "Network Load", value: `${(((networkStats.totalPorts - networkStats.totalAvail) / Math.max(networkStats.totalPorts, 1)) * 100).toFixed(0)}%`, sub: "utilisation", color: "#8b5cf6" },
          ].map(({ icon, label, value, sub, color }) => (
            <div className="kpi-card" key={label} style={{ "--accent": color }}>
              <div className="kpi-accent-bar" style={{ background: color }} />
              <div className="kpi-icon">{icon}</div>
              <div className="kpi-value" style={{ color }}>{value}</div>
              <div className="kpi-label">{label}</div>
              <div className="kpi-sub">{sub}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Main grid ──────────────────────────────────────────────────────── */}
      <div className="dashboard-grid">

        {/* Left column: Stations list */}
        <div className="dashboard-col-main">
          <div className="card">
            <div className="card-header">
              <h2>Station Status</h2>
              <div className="sort-controls">
                <span className="sort-label">Sort:</span>
                {["wait", "distance", "availability"].map((s) => (
                  <button
                    key={s}
                    className={`sort-btn ${sortBy === s ? "active" : ""}`}
                    onClick={() => setSortBy(s)}
                  >
                    {s === "wait" ? "⏱ Wait" : s === "distance" ? "📍 Distance" : "🟢 Ports"}
                  </button>
                ))}
              </div>
            </div>
            <div className="station-grid">
              {sortedStations.map((station) => (
                <StationCard
                  key={station.station_id}
                  station={station}
                  compact
                  onClick={() => navigate(`/stations/${station.station_id}`)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right column: Forecast + Queue + Recommendations */}
        <div className="dashboard-col-side">

          {/* Forecast chart */}
          <div className="card">
            <div className="card-header">
              <h2>24h Demand Forecast</h2>
              <select
                value={selectedForecastStation}
                onChange={(e) => setSelectedForecastStation(Number(e.target.value))}
                className="select-sm"
              >
                {stations.slice(0, 8).map((s) => (
                  <option key={s.station_id} value={s.station_id}>
                    {s.name?.split("–")[0].trim()}
                  </option>
                ))}
              </select>
            </div>
            <ForecastChart data={forecast} currentHour={new Date().getHours()} />
          </div>

          {/* Queue gauge for top busy station */}
          {stations.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h2>Queue Analysis (M/M/c)</h2>
              </div>
              <QueueGauge
                station={stations.reduce((a, b) =>
                  (a.queue_size || 0) > (b.queue_size || 0) ? a : b
                )}
                API_BASE={API_BASE}
              />
            </div>
          )}

          {/* Recommendations */}
          <div className="card">
            <div className="card-header">
              <h2>📍 Nearest Available</h2>
            </div>
            <RecommendationList
              userLocation={userLocation}
              API_BASE={API_BASE}
              topN={4}
            />
          </div>

        </div>
      </div>
    </div>
  );
}

// ─── Loading Screen ───────────────────────────────────────────────────────────
function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-spinner">⚡</div>
      <p>Loading EV ChargeSmart...</p>
    </div>
  );
}

// ─── Mock forecast (API offline) ─────────────────────────────────────────────
function generateMockForecast() {
  return Array.from({ length: 12 }, (_, i) => {
    const h = (new Date().getHours() + i) % 24;
    const base = h >= 7 && h <= 9 ? 28 : h >= 17 && h <= 20 ? 35 : h < 6 ? 5 : 18;
    return {
      hour: h,
      label: `${String(h).padStart(2, "0")}:00`,
      predicted_wait_min: base + (Math.random() - 0.5) * 6,
      recommendation: base < 10 ? "GO_NOW" : base < 20 ? "GOOD_TIME" : "MODERATE_WAIT",
    };
  });
}
