// frontend/src/components/StationCard.js
// Reusable station info card — compact (list view) and full (detail panel).

import React from "react";
import WaitTimeBadge from "./WaitTimeBadge";

const STATUS_COLORS = {
  Operational: "#22c55e",
  Partial: "#f59e0b",
  Offline: "#ef4444",
};

const CONNECTOR_ICONS = {
  CCS: "🔵", CHAdeMO: "🟡", "Type 2": "🟢", Tesla: "🔴", Unknown: "⚪",
};

export default function StationCard({ station, prediction, compact = true, onClick }) {
  const wait = prediction?.predicted_wait_min ?? station.wait_time_minutes ?? station.predicted_wait_min ?? 20;
  const avail = station.available_ports ?? 0;
  const total = station.num_ports ?? 1;
  const utilPct = Math.round(((total - avail) / Math.max(total, 1)) * 100);
  const statusColor = STATUS_COLORS[station.status] || "#64748b";
  const connIcon = CONNECTOR_ICONS[station.connector_type] || "⚪";

  if (compact) {
    return (
      <div
        className="station-card-compact"
        onClick={onClick}
        style={{ borderLeft: `3px solid ${statusColor}` }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && onClick?.()}
      >
        <div className="sc-header">
          <span className="sc-name">{station.name}</span>
          <WaitTimeBadge wait={wait} />
        </div>
        <div className="sc-meta">
          <span>{connIcon} {station.connector_type}</span>
          <span>⚡ {station.power_kw ?? "?"}kW</span>
          <span>🟢 {avail}/{total}</span>
          <span>🚗 Queue: {station.queue_size ?? 0}</span>
        </div>
        <div className="sc-util-bar">
          <div
            className="sc-util-fill"
            style={{
              width: `${utilPct}%`,
              background: utilPct > 80 ? "#ef4444" : utilPct > 60 ? "#f59e0b" : "#22c55e",
            }}
          />
        </div>
        <div className="sc-util-label">{utilPct}% utilised</div>
      </div>
    );
  }

  // Full card
  return (
    <div className="station-card-full" style={{ borderTop: `3px solid ${statusColor}` }}>
      <div className="scf-header">
        <div>
          <h3 className="scf-name">{station.name}</h3>
          <p className="scf-operator">{station.operator}</p>
        </div>
        <span
          className="scf-status-badge"
          style={{ background: `${statusColor}22`, color: statusColor, border: `1px solid ${statusColor}44` }}
        >
          {station.status}
        </span>
      </div>

      <div className="scf-stats">
        {[
          { label: "Predicted Wait", value: `${Math.round(wait)} min`, color: wait < 15 ? "#22c55e" : wait < 30 ? "#f59e0b" : "#ef4444" },
          { label: "Available Ports", value: `${avail}/${total}`, color: "#22c55e" },
          { label: "Queue Size", value: station.queue_size ?? 0, color: "#f59e0b" },
          { label: "Traffic Score", value: parseFloat(station.traffic_score ?? 0).toFixed(2), color: "#3b82f6" },
        ].map(({ label, value, color }) => (
          <div className="scf-stat" key={label}>
            <div className="scf-stat-value" style={{ color }}>{value}</div>
            <div className="scf-stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div className="scf-details">
        <div className="scf-detail-row">
          <span>{connIcon} {station.connector_type}</span>
          <span>⚡ {station.power_kw ?? "?"}kW</span>
          {station.distance_km && <span>📍 {station.distance_km?.toFixed(1)} km</span>}
        </div>
      </div>

      <div className="scf-util-section">
        <div className="scf-util-label">Station Utilisation: {utilPct}%</div>
        <div className="scf-util-bar">
          <div
            className="scf-util-fill"
            style={{
              width: `${utilPct}%`,
              background: utilPct > 80 ? "#ef4444" : utilPct > 60 ? "#f59e0b" : "#22c55e",
            }}
          />
        </div>
      </div>

      {prediction && (
        <div className="scf-prediction">
          <div className="scf-pred-label">ML Ensemble Prediction</div>
          <div className="scf-pred-row">
            <span>🌲 RF: {prediction.rf_prediction ?? "N/A"} min</span>
            <span>📐 Queue: {prediction.queue_prediction?.toFixed(1)} min</span>
            <span className={`scf-rec rec-${prediction.recommendation}`}>
              {prediction.recommendation?.replace(/_/g, " ")}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
