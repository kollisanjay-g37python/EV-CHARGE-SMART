// frontend/src/components/RecommendationList.js
// Shows ranked station recommendations from /api/v1/recommend.

import React, { useState, useEffect } from "react";
import { fetchRecommendations } from "../services/api";
import WaitTimeBadge from "./WaitTimeBadge";

const ACTION_STYLE = {
  GO_NOW:        { color: "#22c55e", bg: "#22c55e22", border: "#22c55e44" },
  RECOMMENDED:   { color: "#00d4aa", bg: "#00d4aa22", border: "#00d4aa44" },
  MODERATE_WAIT: { color: "#f59e0b", bg: "#f59e0b22", border: "#f59e0b44" },
  LONG_WAIT:     { color: "#ef4444", bg: "#ef444422", border: "#ef444444" },
  HIGH_WAIT:     { color: "#ef4444", bg: "#ef444422", border: "#ef444444" },
};

export default function RecommendationList({
  userLocation,
  API_BASE,
  topN = 5,
  priority = "balanced",
  connectorType = null,
}) {
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activePriority, setActivePriority] = useState(priority);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchRecommendations(API_BASE, {
          user_lat: userLocation.lat,
          user_lng: userLocation.lng,
          priority: activePriority,
          connector_type: connectorType,
          top_n: topN,
        });
        setRecs(data.recommendations || []);
      } catch (err) {
        setError("Could not load recommendations");
        setRecs(generateMockRecs(topN));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [userLocation, activePriority, connectorType, topN, API_BASE]);

  return (
    <div className="recommendation-list">
      {/* Priority toggles */}
      <div className="priority-toggles">
        {["balanced", "speed", "distance", "availability"].map((p) => (
          <button
            key={p}
            className={`priority-btn ${activePriority === p ? "active" : ""}`}
            onClick={() => setActivePriority(p)}
          >
            {p === "speed" ? "⚡ Speed" : p === "distance" ? "📍 Distance" : p === "availability" ? "🟢 Ports" : "⚖ Balanced"}
          </button>
        ))}
      </div>

      {loading && <div className="rec-loading">Finding best stations...</div>}
      {error && <div className="rec-error">{error} (showing demo data)</div>}

      <div className="rec-items">
        {recs.map((rec, idx) => {
          const actionKey = rec.action?.replace(/[⚡✅⏳🔴]\s*/g, "").replace(/ /g, "_").toUpperCase();
          const style = ACTION_STYLE[actionKey] || ACTION_STYLE.MODERATE_WAIT;

          return (
            <div className="rec-card" key={rec.station_id || idx}>
              {/* Rank badge */}
              <div className="rec-rank-badge">#{rec.rank || idx + 1}</div>

              <div className="rec-body">
                <div className="rec-name">{rec.name}</div>
                <div className="rec-meta">
                  <span>📍 {rec.distance_km?.toFixed(1)} km</span>
                  <span>🔌 {rec.connector_type}</span>
                  <span>⚡ {rec.power_kw?.toFixed(0)}kW</span>
                  <span>🟢 {rec.available_ports} free</span>
                </div>

                {/* Score bar */}
                <div className="rec-score-row">
                  <span className="rec-score-label">Score</span>
                  <div className="rec-score-bar">
                    <div
                      className="rec-score-fill"
                      style={{ width: `${(rec.composite_score || 0) * 100}%` }}
                    />
                  </div>
                  <span className="rec-score-val">
                    {((rec.composite_score || 0) * 100).toFixed(0)}
                  </span>
                </div>
              </div>

              <div className="rec-right">
                <WaitTimeBadge wait={rec.predicted_wait_min} />
                <span
                  className="rec-action-badge"
                  style={{ color: style.color, background: style.bg, border: `1px solid ${style.border}` }}
                >
                  {rec.action}
                </span>
                {rec.routing_url && (
                  <a href={rec.routing_url} target="_blank" rel="noreferrer" className="rec-nav-link">
                    🗺 Go
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function generateMockRecs(n) {
  const names = [
    "Tesla Supercharger – Downtown Hub",
    "ChargePoint – Westfield Mall",
    "EVgo – Airport Terminal B",
    "Blink – City Center Plaza",
    "Electrify America – Oak Grove",
  ];
  return Array.from({ length: Math.min(n, names.length) }, (_, i) => ({
    station_id: i + 1,
    rank: i + 1,
    name: names[i],
    distance_km: 0.8 + i * 0.9,
    predicted_wait_min: 5 + i * 8,
    available_ports: Math.max(0, 4 - i),
    connector_type: ["CCS", "Type 2", "CHAdeMO", "Tesla", "CCS"][i],
    power_kw: [150, 50, 350, 150, 50][i],
    composite_score: 0.92 - i * 0.1,
    action: i === 0 ? "⚡ GO NOW" : i < 2 ? "✅ RECOMMENDED" : "⏳ MODERATE WAIT",
    routing_url: `https://www.google.com/maps/dir/?api=1&destination=37.77${i},-122.42`,
  }));
}
