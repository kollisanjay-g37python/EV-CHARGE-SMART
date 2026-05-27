// frontend/src/App.js
// Root React component — sets up routing, global state, and layout.
// API base: http://localhost:8000/api/v1  (FastAPI backend)

import React, { useState, useEffect, createContext, useContext } from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import MapView from "./components/MapView";
import StationDetail from "./components/StationDetail";
import PredictPage from "./components/PredictPage";
import RecommendPage from "./components/RecommendPage";
import AnalyticsPage from "./components/AnalyticsPage";
import "./App.css";

// ─── Global Context ───────────────────────────────────────────────────────────
export const AppContext = createContext();

export const useApp = () => useContext(AppContext);

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

function App() {
  const [stations, setStations] = useState([]);
  const [userLocation, setUserLocation] = useState({ lat: 37.7749, lng: -122.4194 });
  const [networkStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [darkMode] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);

  // ─── Fetch stations on mount ──────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      try {
        // Health check
        const health = await fetch(`${API_BASE.replace("/api/v1", "")}/health`);
        setApiOnline(health.ok);

        // Stations
        const resp = await fetch(
          `${API_BASE}/stations?lat=${userLocation.lat}&lng=${userLocation.lng}&limit=20`
        );
        if (resp.ok) {
          const data = await resp.json();
          setStations(data.stations);
        }
      } catch (err) {
        console.warn("API offline — using mock data", err);
        setApiOnline(false);
        setStations(generateMockStations());
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [userLocation]);

  // ─── Auto-refresh every 60s ───────────────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${API_BASE}/stations?limit=20`);
        if (resp.ok) {
          const data = await resp.json();
          setStations(data.stations);
        }
      } catch {}
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  // ─── Geolocation ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        () => {}   // fallback to default
      );
    }
  }, []);

  const contextValue = {
    stations, setStations,
    userLocation, setUserLocation,
    networkStatus, loading,
    apiOnline, API_BASE,
  };

  return (
    <AppContext.Provider value={contextValue}>
      <Router>
        <div className={`app ${darkMode ? "dark" : ""}`}>
          <NavBar apiOnline={apiOnline} />
          <main className="app-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/map" element={<MapView />} />
              <Route path="/stations/:id" element={<StationDetail />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/recommend" element={<RecommendPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
            </Routes>
          </main>
          <StatusBar apiOnline={apiOnline} stationCount={stations.length} />
        </div>
      </Router>
    </AppContext.Provider>
  );
}

// ─── Nav Bar ──────────────────────────────────────────────────────────────────
function NavBar({ apiOnline }) {
  const navItems = [
    { to: "/", label: "📊 Dashboard", exact: true },
    { to: "/map", label: "🗺️ Map" },
    { to: "/predict", label: "⚡ Predict" },
    { to: "/recommend", label: "📍 Recommend" },
    { to: "/analytics", label: "📈 Analytics" },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="brand-icon">⚡</span>
        <span className="brand-name">EV ChargeSmart</span>
        <span className="brand-sub">Wait-Time Prediction</span>
      </div>
      <ul className="navbar-links">
        {navItems.map(({ to, label, exact }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={exact}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="navbar-status">
        <span className={`api-badge ${apiOnline ? "online" : "offline"}`}>
          <span className="status-dot" />
          {apiOnline ? "API Live" : "API Offline"}
        </span>
      </div>
    </nav>
  );
}

// ─── Status Bar ───────────────────────────────────────────────────────────────
function StatusBar({ apiOnline, stationCount }) {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <footer className="status-bar">
      <span>⚡ EV ChargeSmart v1.0</span>
      <span>{stationCount} stations tracked</span>
      <span>Models: RF + LSTM + M/M/c Queue</span>
      <span className="status-time">{time.toLocaleTimeString()}</span>
    </footer>
  );
}

// ─── Mock data (API offline fallback) ────────────────────────────────────────
function generateMockStations() {
  const names = [
    "Tesla Supercharger – Downtown Hub",
    "ChargePoint – Westfield Mall",
    "EVgo – Airport Terminal B",
    "Blink – City Center Plaza",
    "Shell Recharge – Highway 101",
    "Electrify America – Oak Grove",
    "ChargePoint – University Ave",
    "EVgo – Harbor District",
  ];
  return names.map((name, i) => ({
    station_id: i + 1, name,
    lat: 37.77 + (Math.random() - 0.5) * 0.1,
    lng: -122.42 + (Math.random() - 0.5) * 0.15,
    num_ports: 4 + Math.floor(Math.random() * 8),
    available_ports: Math.floor(Math.random() * 5),
    queue_size: Math.floor(Math.random() * 6),
    connector_type: ["CCS", "CHAdeMO", "Type 2", "Tesla"][i % 4],
    power_kw: [7.2, 50, 150, 350][i % 4],
    status: ["Operational", "Operational", "Partial", "Offline"][Math.floor(Math.random() * 4)],
    operator: ["Tesla", "ChargePoint", "EVgo", "Blink"][i % 4],
    traffic_score: Math.random().toFixed(2),
    wait_time_minutes: Math.floor(Math.random() * 45),
  }));
}

export default App;
