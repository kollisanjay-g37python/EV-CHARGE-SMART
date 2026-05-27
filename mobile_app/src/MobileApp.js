// mobile_app/src/MobileApp.js
// Capacitor-wrapped React Native-style mobile interface
// Run: npm run build && npx cap sync && npx cap open android

import React, { useState, useEffect, useCallback } from "react";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const C = {
  bg:     "#060a10",
  panel:  "#0d1520",
  card:   "#111d2c",
  border: "#1e2d3d",
  green:  "#00ff88",
  blue:   "#00aaff",
  orange: "#ff6b35",
  yellow: "#ffd60a",
  text:   "#d0e8ff",
  muted:  "#4a6480",
};

const AVAIL = {
  Available: C.green, Moderate: C.yellow, Busy: C.orange, Full: "#ff3333",
};

const css = {
  root: {
    background: C.bg, minHeight: "100vh", color: C.text,
    fontFamily: "'DM Sans', -apple-system, sans-serif",
    maxWidth: "430px", margin: "0 auto", position: "relative",
    overflow: "hidden",
  },
  statusBar: {
    background: C.panel, padding: "12px 20px 8px",
    display: "flex", justifyContent: "space-between", alignItems: "center",
    borderBottom: `1px solid ${C.border}`,
  },
  appTitle: {
    fontFamily: "monospace", color: C.green, fontWeight: "700",
    fontSize: "0.9rem", letterSpacing: "-0.5px",
  },
  statusPill: (ok) => ({
    display: "flex", alignItems: "center", gap: "5px",
    background: ok ? "rgba(0,255,136,0.1)" : "rgba(255,107,53,0.1)",
    border: `1px solid ${ok ? C.green : C.orange}33`,
    padding: "3px 10px", borderRadius: "99px",
    fontSize: "0.65rem", color: ok ? C.green : C.orange, fontFamily: "monospace",
  }),
  liveDot: {
    width: "6px", height: "6px", borderRadius: "50%",
    background: C.green, animation: "pulse 2s ease infinite",
  },
  heroCard: {
    margin: "16px", padding: "20px",
    background: `linear-gradient(135deg, ${C.card}, ${C.panel})`,
    border: `1px solid ${C.border}`, borderRadius: "16px",
    position: "relative", overflow: "hidden",
  },
  heroGlow: {
    position: "absolute", top: "-40px", right: "-40px",
    width: "150px", height: "150px", borderRadius: "50%",
    background: `radial-gradient(circle, ${C.green}22, transparent 70%)`,
    pointerEvents: "none",
  },
  heroLabel: { fontSize: "0.65rem", color: C.muted, fontFamily: "monospace",
                textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "6px" },
  heroWait: (color) => ({
    fontSize: "3.5rem", fontWeight: "800", color, fontFamily: "monospace",
    lineHeight: 1, textShadow: `0 0 30px ${color}55`,
  }),
  heroUnit: { fontSize: "1rem", fontWeight: "400", color: C.muted, marginLeft: "4px" },
  heroMeta: { display: "flex", gap: "16px", marginTop: "12px", fontSize: "0.78rem", color: C.muted },
  heroBadge: (avail) => ({
    display: "inline-block", padding: "3px 10px", borderRadius: "99px",
    fontSize: "0.65rem", fontWeight: "700", fontFamily: "monospace",
    background: `${AVAIL[avail] || C.muted}22`, color: AVAIL[avail] || C.muted,
    border: `1px solid ${AVAIL[avail] || C.muted}44`,
    marginTop: "10px",
  }),
  quickStats: {
    display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
    gap: "10px", margin: "0 16px 16px",
  },
  statCard: {
    background: C.card, border: `1px solid ${C.border}`,
    borderRadius: "12px", padding: "14px 12px", textAlign: "center",
  },
  statVal: (color) => ({
    fontFamily: "monospace", fontSize: "1.4rem", fontWeight: "700",
    color, lineHeight: 1,
  }),
  statLabel: { fontSize: "0.6rem", color: C.muted, marginTop: "4px",
                textTransform: "uppercase", letterSpacing: "0.06em" },
  sectionLabel: {
    fontFamily: "monospace", fontSize: "0.65rem", color: C.green,
    letterSpacing: "0.12em", textTransform: "uppercase",
    padding: "0 16px", marginBottom: "8px",
    display: "flex", justifyContent: "space-between", alignItems: "center",
  },
  stationCard: (selected) => ({
    margin: "0 16px 8px",
    background: C.card, border: `1px solid ${selected ? C.green : C.border}`,
    borderRadius: "12px", padding: "14px 16px", cursor: "pointer",
    transition: "border-color 0.15s, transform 0.1s",
    transform: selected ? "scale(1.01)" : "scale(1)",
  }),
  stationRow: {
    display: "flex", justifyContent: "space-between", alignItems: "flex-start",
  },
  stationName: { fontWeight: "600", fontSize: "0.9rem", marginBottom: "5px" },
  stationMeta: {
    fontSize: "0.7rem", color: C.muted, fontFamily: "monospace",
    display: "flex", gap: "10px", flexWrap: "wrap",
  },
  waitChip: (color) => ({
    fontFamily: "monospace", fontSize: "1.1rem", fontWeight: "700", color,
    textShadow: `0 0 12px ${color}44`,
  }),
  availBadge: (avail) => ({
    fontSize: "0.6rem", fontWeight: "700", fontFamily: "monospace",
    color: AVAIL[avail] || C.muted, marginTop: "4px",
    display: "block", textAlign: "right",
  }),
  progressBar: (pct, color) => ({
    height: "3px", borderRadius: "99px",
    background: C.border, marginTop: "10px", overflow: "hidden",
  }),
  progressFill: (pct, color) => ({
    height: "100%", width: `${pct}%`, background: color,
    borderRadius: "99px", transition: "width 0.6s ease",
  }),
  bottomNav: {
    position: "fixed", bottom: 0, left: "50%",
    transform: "translateX(-50%)", width: "100%", maxWidth: "430px",
    background: C.panel, borderTop: `1px solid ${C.border}`,
    padding: "10px 0 20px",
    display: "flex", justifyContent: "space-around",
  },
  navItem: (active) => ({
    display: "flex", flexDirection: "column", alignItems: "center",
    gap: "3px", cursor: "pointer", padding: "4px 16px",
    color: active ? C.green : C.muted,
    fontSize: "0.62rem", fontFamily: "monospace",
    transition: "color 0.15s",
  }),
  navIcon: { fontSize: "1.3rem" },
  refreshBtn: {
    fontSize: "0.65rem", color: C.green, fontFamily: "monospace",
    background: "none", border: `1px solid ${C.green}44`,
    borderRadius: "6px", padding: "3px 8px", cursor: "pointer",
  },
};

function StationCard({ station, selected, onClick }) {
  const avail = station.availability || "Moderate";
  const color = AVAIL[avail] || C.muted;
  const wait  = station.blended_wait_min ?? station.wait_min ?? "?";
  const pct   = Math.min(100, (Number(wait) / 30) * 100);

  return (
    <div style={css.stationCard(selected)} onClick={() => onClick(station)}>
      <div style={css.stationRow}>
        <div>
          <div style={css.stationName}>{station.name || `Station #${station.station_id}`}</div>
          <div style={css.stationMeta}>
            {station.distance_km != null && <span>📍 {Number(station.distance_km).toFixed(1)} km</span>}
            {station.num_connectors && <span>🔌 {station.num_connectors}</span>}
            {station.is_free && <span style={{ color: C.green }}>🆓</span>}
            <span>{station.connector_types?.split(",")[0]}</span>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={css.waitChip(color)}>{wait}<span style={{ fontSize:"0.65rem", color: C.muted }}> min</span></div>
          <span style={css.availBadge(avail)}>{avail}</span>
        </div>
      </div>
      <div style={css.progressBar(pct, color)}>
        <div style={css.progressFill(pct, color)}/>
      </div>
    </div>
  );
}

export default function MobileApp() {
  const [tab,           setTab]          = useState("home");
  const [stations,      setStations]     = useState([]);
  const [selectedSt,    setSelectedSt]   = useState(null);
  const [apiOk,         setApiOk]        = useState(false);
  const [loading,       setLoading]      = useState(false);
  const LAT = 37.7749, LON = -122.4194;

  const rng = s => { let x = Math.sin(s * 9999) * 9999; return x - Math.floor(x); };

  const loadStations = useCallback(async () => {
    setLoading(true);
    try {
      const res  = await fetch(`${API}/stations?lat=${LAT}&lon=${LON}&radius_km=10`);
      const json = await res.json();
      setStations(json.stations || []);
      setApiOk(true);
    } catch {
      const avails = ["Available","Available","Moderate","Busy","Full","Available","Moderate"];
      setStations(Array.from({ length: 8 }, (_, i) => ({
        station_id: i+1,
        name: ["Downtown FastCharge","SoMa Hub","Market St EV","Embarcadero Pt",
               "Caltrain Plaza","Union Sq Charge","Castro EV","Mission Bay"][i],
        distance_km: +(rng(i*3)*7+0.3).toFixed(1),
        blended_wait_min: Math.round(rng(i*7)*25+2),
        num_connectors: Math.round(rng(i*11)*10+2),
        availability: avails[i],
        connector_types: ["CCS,CHAdeMO","Tesla","J1772","CCS","CHAdeMO","Tesla","CCS","J1772"][i],
        is_free: rng(i*13)>0.7,
      })));
    }
    setLoading(false);
  }, []);

  useEffect(() => { loadStations(); }, [loadStations]);

  const nearest   = [...stations].sort((a,b)=>(a.distance_km||99)-(b.distance_km||99))[0];
  const available = stations.filter(s=>s.availability==="Available").length;
  const avgWait   = stations.length
    ? Math.round(stations.reduce((a,s)=>a+(s.blended_wait_min||10),0)/stations.length)
    : "—";

  const navItems = [
    { id:"home",   icon:"🏠", label:"Home"    },
    { id:"map",    icon:"🗺️",  label:"Map"     },
    { id:"search", icon:"🔍", label:"Find"    },
    { id:"alerts", icon:"🔔", label:"Alerts"  },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        ::-webkit-scrollbar { display:none; }
      `}</style>

      <div style={css.root}>
        {/* Status bar */}
        <div style={css.statusBar}>
          <div style={css.appTitle}>⚡ EV SMART</div>
          <div style={css.statusPill(apiOk)}>
            {apiOk && <div style={css.liveDot}/>}
            {apiOk ? "LIVE" : "DEMO"}
          </div>
        </div>

        {/* Scrollable body */}
        <div style={{ overflowY:"auto", paddingBottom:"90px" }}>

          {/* Hero card — nearest station */}
          {nearest && (
            <div style={css.heroCard}>
              <div style={css.heroGlow}/>
              <div style={css.heroLabel}>Nearest station</div>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
                <div>
                  <div style={{ fontWeight:"700", fontSize:"1.05rem", marginBottom:"4px" }}>
                    {nearest.name}
                  </div>
                  <div style={{ fontSize:"0.72rem", color:C.muted, fontFamily:"monospace" }}>
                    📍 {nearest.distance_km} km away
                  </div>
                  <span style={css.heroBadge(nearest.availability)}>{nearest.availability}</span>
                </div>
                <div>
                  <span style={css.heroWait(AVAIL[nearest.availability]||C.muted)}>
                    {nearest.blended_wait_min}
                    <span style={css.heroUnit}>min</span>
                  </span>
                </div>
              </div>
              <div style={css.heroMeta}>
                <span>🔌 {nearest.num_connectors} connectors</span>
                <span>{nearest.connector_types?.split(",")[0]}</span>
                {nearest.is_free && <span style={{color:C.green}}>🆓 Free</span>}
              </div>
            </div>
          )}

          {/* Quick stats */}
          <div style={css.quickStats}>
            <div style={css.statCard}>
              <div style={css.statVal(C.green)}>{stations.length}</div>
              <div style={css.statLabel}>Nearby</div>
            </div>
            <div style={css.statCard}>
              <div style={css.statVal(C.green)}>{available}</div>
              <div style={css.statLabel}>Available</div>
            </div>
            <div style={css.statCard}>
              <div style={css.statVal(C.yellow)}>{avgWait}<span style={{fontSize:"0.7rem",color:C.muted}}>m</span></div>
              <div style={css.statLabel}>Avg Wait</div>
            </div>
          </div>

          {/* Station list */}
          <div style={css.sectionLabel}>
            <span>Stations Nearby</span>
            <button style={css.refreshBtn} onClick={loadStations} disabled={loading}>
              {loading ? "…" : "↻ Refresh"}
            </button>
          </div>

          {stations.map(s => (
            <StationCard
              key={s.station_id}
              station={s}
              selected={selectedSt?.station_id === s.station_id}
              onClick={setSelectedSt}
            />
          ))}
        </div>

        {/* Bottom nav */}
        <div style={css.bottomNav}>
          {navItems.map(n => (
            <div key={n.id} style={css.navItem(tab===n.id)} onClick={()=>setTab(n.id)}>
              <span style={css.navIcon}>{n.icon}</span>
              <span>{n.label}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
