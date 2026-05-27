// frontend/src/components/MapView.js
// Interactive Leaflet map with live station markers, clustering,
// and click-to-predict functionality.
// Map tiles: OpenStreetMap (https://www.openstreetmap.org)

import React, { useEffect, useRef, useState, useCallback } from "react";
import { useApp } from "../App";
import { useNavigate } from "react-router-dom";
import { fetchPrediction, fetchRecommendations } from "../services/api";
import WaitTimeBadge from "./WaitTimeBadge";
import StationCard from "./StationCard";

// Leaflet loaded via CDN in index.html
const L = window.L;

// Wait → marker colour
const waitColor = (wait) =>
  wait < 5 ? "#22c55e" : wait < 15 ? "#eab308" : wait < 30 ? "#f97316" : "#ef4444";

const markerIcon = (wait, ports) => {
  const color = waitColor(wait);
  const size = Math.min(32, 18 + ports * 1.2);
  return L &&
    L.divIcon({
      className: "",
      html: `<div style="
        width:${size}px;height:${size}px;border-radius:50%;
        background:${color};border:2px solid #fff;
        box-shadow:0 0 8px ${color}88;
        display:flex;align-items:center;justify-content:center;
        font-size:10px;font-weight:700;color:#000;">
        ${Math.round(wait)}
      </div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
};

export default function MapView() {
  const { stations, userLocation, setUserLocation, API_BASE } = useApp();
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const markersRef = useRef({});
  const navigate = useNavigate();

  const [selectedStation, setSelectedStation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [showRec, setShowRec] = useState(false);
  const [mapStyle, setMapStyle] = useState("dark");
  const [filterStatus, setFilterStatus] = useState("all");

  const TILE_URLS = {
    dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    osm: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  };

  // ─── Init map ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!L || mapInstance.current) return;
    mapInstance.current = L.map(mapRef.current, {
      center: [userLocation.lat, userLocation.lng],
      zoom: 13,
      zoomControl: true,
    });
    L.tileLayer(TILE_URLS[mapStyle], {
      attribution: '© <a href="https://www.openstreetmap.org">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(mapInstance.current);
  }, []);

  // ─── Update tile layer on style change ────────────────────────────────────
  useEffect(() => {
    if (!mapInstance.current || !L) return;
    mapInstance.current.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) mapInstance.current.removeLayer(layer);
    });
    L.tileLayer(TILE_URLS[mapStyle], {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(mapInstance.current);
  }, [mapStyle]);

  // ─── User location marker ─────────────────────────────────────────────────
  useEffect(() => {
    if (!mapInstance.current || !L) return;
    const icon = L.divIcon({
      className: "",
      html: `<div style="width:16px;height:16px;border-radius:50%;background:#3b82f6;
             border:3px solid #fff;box-shadow:0 0 12px #3b82f688;"></div>`,
      iconSize: [16, 16], iconAnchor: [8, 8],
    });
    L.marker([userLocation.lat, userLocation.lng], { icon })
      .addTo(mapInstance.current)
      .bindPopup("<b>📍 Your Location</b>");
  }, [userLocation]);

  // ─── Station markers ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapInstance.current || !L) return;

    // Clear old markers
    Object.values(markersRef.current).forEach((m) => mapInstance.current.removeLayer(m));
    markersRef.current = {};

    const filtered = filterStatus === "all"
      ? stations
      : stations.filter((s) => s.status?.toLowerCase().includes(filterStatus));

    filtered.forEach((station) => {
      const wait = station.wait_time_minutes ?? station.predicted_wait_min ?? 20;
      const icon = markerIcon(wait, station.num_ports || 4);
      if (!icon) return;

      const marker = L.marker([station.lat, station.lng], { icon })
        .addTo(mapInstance.current)
        .bindTooltip(
          `<b>${station.name}</b><br>Wait: ${Math.round(wait)} min | ${station.available_ports}/${station.num_ports} ports`,
          { direction: "top", offset: [0, -10] }
        )
        .on("click", () => handleStationClick(station));

      markersRef.current[station.station_id] = marker;
    });
  }, [stations, filterStatus]);

  // ─── Click handler ────────────────────────────────────────────────────────
  const handleStationClick = useCallback(async (station) => {
    setSelectedStation(station);
    setPrediction(null);
    try {
      const pred = await fetchPrediction(API_BASE, {
        station_id: station.station_id,
        num_ports: station.num_ports,
        available_ports: station.available_ports,
        queue_size: station.queue_size,
        hour: new Date().getHours(),
        day_of_week: new Date().getDay(),
        traffic_score: parseFloat(station.traffic_score) || 0.5,
      });
      setPrediction(pred);
    } catch {}
  }, [API_BASE]);

  // ─── Recommendations ──────────────────────────────────────────────────────
  const loadRecommendations = async () => {
    try {
      const recs = await fetchRecommendations(API_BASE, {
        user_lat: userLocation.lat,
        user_lng: userLocation.lng,
        priority: "balanced",
        top_n: 5,
      });
      setRecommendations(recs.recommendations || []);
      setShowRec(true);
    } catch {}
  };

  return (
    <div className="mapview-container">
      {/* Controls */}
      <div className="map-controls">
        <div className="map-control-group">
          <label>Map Style</label>
          <select value={mapStyle} onChange={(e) => setMapStyle(e.target.value)}>
            <option value="dark">Dark</option>
            <option value="light">Light</option>
            <option value="osm">OpenStreetMap</option>
          </select>
        </div>
        <div className="map-control-group">
          <label>Filter</label>
          <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
            <option value="all">All Stations</option>
            <option value="operational">Operational</option>
            <option value="partial">Partial</option>
          </select>
        </div>
        <button className="btn-primary" onClick={loadRecommendations}>
          📍 Get Recommendations
        </button>
      </div>

      {/* Legend */}
      <div className="map-legend">
        {[
          { color: "#22c55e", label: "< 5 min" },
          { color: "#eab308", label: "5–15 min" },
          { color: "#f97316", label: "15–30 min" },
          { color: "#ef4444", label: "> 30 min" },
        ].map(({ color, label }) => (
          <div key={label} className="legend-item">
            <span className="legend-dot" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Map Container */}
      <div ref={mapRef} className="leaflet-map" />

      {/* Station Panel */}
      {selectedStation && (
        <div className="station-panel">
          <button className="panel-close" onClick={() => setSelectedStation(null)}>✕</button>
          <StationCard station={selectedStation} prediction={prediction} compact={false} />
          <div className="panel-actions">
            <button
              className="btn-primary"
              onClick={() => navigate(`/stations/${selectedStation.station_id}`)}
            >
              View Details
            </button>
            <a
              className="btn-secondary"
              href={`https://www.google.com/maps/dir/?api=1&destination=${selectedStation.lat},${selectedStation.lng}`}
              target="_blank" rel="noreferrer"
            >
              🗺 Navigate
            </a>
          </div>
        </div>
      )}

      {/* Recommendations Panel */}
      {showRec && recommendations.length > 0 && (
        <div className="rec-panel">
          <div className="rec-panel-header">
            <h3>📍 Top Recommendations</h3>
            <button onClick={() => setShowRec(false)}>✕</button>
          </div>
          {recommendations.map((rec) => (
            <div key={rec.station_id} className="rec-item"
              onClick={() => {
                if (mapInstance.current) mapInstance.current.flyTo([rec.lat, rec.lng], 15);
                setShowRec(false);
              }}>
              <span className="rec-rank">#{rec.rank}</span>
              <div className="rec-info">
                <div className="rec-name">{rec.name}</div>
                <div className="rec-meta">
                  📍 {rec.distance_km?.toFixed(1)} km · ⏱ {Math.round(rec.predicted_wait_min)} min
                </div>
              </div>
              <WaitTimeBadge wait={rec.predicted_wait_min} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
