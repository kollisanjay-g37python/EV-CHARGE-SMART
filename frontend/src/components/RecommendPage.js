// frontend/src/components/RecommendPage.js
// Full recommendation page with priority controls, connector filter,
// map preview, and ranked station cards.

import React, { useState } from "react";
import { useApp } from "../App";
import RecommendationList from "./RecommendationList";

const PRIORITIES = [
  { key: "balanced",     icon: "⚖",  label: "Balanced" },
  { key: "speed",        icon: "⚡",  label: "Shortest Wait" },
  { key: "distance",     icon: "📍",  label: "Nearest" },
  { key: "availability", icon: "🟢",  label: "Most Available" },
];
const CONNECTORS = ["Any", "CCS", "CHAdeMO", "Type 2", "Tesla"];

export default function RecommendPage() {
  const { userLocation, setUserLocation, API_BASE } = useApp();

  const [priority, setPriority]       = useState("balanced");
  const [connector, setConnector]     = useState("Any");
  const [topN, setTopN]               = useState(5);
  const [detour, setDetour]           = useState(20);
  const [targetHour, setTargetHour]   = useState("");

  return (
    <div className="recommend-page">
      <div className="page-header">
        <h1 className="page-title">📍 Smart Recommendations</h1>
        <p className="page-subtitle">
          Multi-factor scoring: wait time · distance · availability · traffic · reliability
        </p>
      </div>

      <div className="recommend-layout">
        {/* Controls */}
        <div className="recommend-controls card">
          <h2>Preferences</h2>

          <div className="form-group">
            <label>Your Location</label>
            <div className="form-row">
              <input type="number" step="0.0001" value={userLocation.lat}
                onChange={(e) => setUserLocation((l) => ({ ...l, lat: Number(e.target.value) }))}
                placeholder="Latitude" />
              <input type="number" step="0.0001" value={userLocation.lng}
                onChange={(e) => setUserLocation((l) => ({ ...l, lng: Number(e.target.value) }))}
                placeholder="Longitude" />
            </div>
            <button className="btn-secondary btn-sm" onClick={() => {
              navigator.geolocation?.getCurrentPosition((pos) =>
                setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude })
              );
            }}>📍 Use My Location</button>
          </div>

          <div className="form-group">
            <label>Optimize For</label>
            <div className="priority-grid">
              {PRIORITIES.map(({ key, icon, label }) => (
                <button
                  key={key}
                  className={`priority-card ${priority === key ? "active" : ""}`}
                  onClick={() => setPriority(key)}
                >
                  <span className="pc-icon">{icon}</span>
                  <span className="pc-label">{label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label>Connector Type</label>
            <div className="connector-btns">
              {CONNECTORS.map((c) => (
                <button
                  key={c}
                  className={`connector-btn ${connector === c ? "active" : ""}`}
                  onClick={() => setConnector(c)}
                >{c}</button>
              ))}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Max Detour: {detour} km</label>
              <input type="range" min={1} max={50} value={detour}
                onChange={(e) => setDetour(Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Results</label>
              <select value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
                {[3, 5, 8, 10].map((n) => <option key={n} value={n}>{n} stations</option>)}
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Future Arrival Hour (optional)</label>
            <select value={targetHour} onChange={(e) => setTargetHour(e.target.value)}>
              <option value="">Now (current time)</option>
              {Array.from({ length: 24 }, (_, h) => (
                <option key={h} value={h}>
                  {String(h).padStart(2, "0")}:00
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Results */}
        <div className="recommend-results card">
          <h2>Recommended Stations</h2>
          <RecommendationList
            userLocation={userLocation}
            API_BASE={API_BASE}
            topN={topN}
            priority={priority}
            connectorType={connector === "Any" ? null : connector}
          />
        </div>
      </div>
    </div>
  );
}
