// frontend/src/components/RecommendationPanel.js
import React, { useState, useCallback } from "react";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const COLORS = {
  bg: "#0d1117", panel: "#161b22", border: "#21262d",
  green: "#39d353", blue: "#58a6ff", orange: "#f78166",
  yellow: "#e3b341", text: "#c9d1d9", muted: "#8b949e",
};
const AVAIL_COLORS = {
  Available: "#39d353", Moderate: "#e3b341",
  Busy: "#f78166", Full: "#da3633",
};
const CONNECTORS = ["Any", "CCS", "CHAdeMO", "Tesla", "J1772"];

const css = {
  wrapper: {
    background: COLORS.panel, border: `1px solid ${COLORS.border}`,
    borderRadius: "12px", overflow: "hidden",
  },
  header: {
    padding: "14px 18px", borderBottom: `1px solid ${COLORS.border}`,
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.72rem", color: COLORS.green,
    letterSpacing: "0.1em", textTransform: "uppercase",
  },
  body: { padding: "16px" },
  filterGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px",
    marginBottom: "12px",
  },
  label: { fontSize: "0.68rem", color: COLORS.muted, marginBottom: "4px", display: "block" },
  select: {
    width: "100%", background: COLORS.bg,
    border: `1px solid ${COLORS.border}`, borderRadius: "8px",
    padding: "7px 10px", color: COLORS.text,
    fontSize: "0.8rem", outline: "none", cursor: "pointer",
  },
  input: {
    width: "100%", background: COLORS.bg,
    border: `1px solid ${COLORS.border}`, borderRadius: "8px",
    padding: "7px 10px", color: COLORS.text,
    fontSize: "0.8rem", outline: "none",
    fontFamily: "'Space Mono', monospace",
  },
  checkRow: {
    display: "flex", alignItems: "center", gap: "8px",
    marginBottom: "12px", fontSize: "0.82rem", color: COLORS.muted, cursor: "pointer",
  },
  btn: {
    width: "100%", padding: "10px",
    background: COLORS.green, color: COLORS.bg,
    border: "none", borderRadius: "8px", fontWeight: "700",
    fontSize: "0.88rem", cursor: "pointer",
    fontFamily: "'DM Sans', sans-serif",
  },
  divider: { height: "1px", background: COLORS.border, margin: "14px 0" },
  recCard: (rank, avail) => ({
    background: COLORS.bg,
    border: `1px solid ${COLORS.border}`,
    borderLeft: `3px solid ${AVAIL_COLORS[avail] || COLORS.muted}`,
    borderRadius: "8px", padding: "12px 14px",
    marginBottom: "8px", cursor: "default",
    transition: "border-color 0.15s",
  }),
  recHeader: {
    display: "flex", justifyContent: "space-between",
    alignItems: "center", marginBottom: "6px",
  },
  rankBadge: {
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.62rem", color: COLORS.muted,
  },
  recName: { fontWeight: "600", fontSize: "0.88rem" },
  statusBadge: (avail) => ({
    display: "inline-block", padding: "2px 8px", borderRadius: "20px",
    fontSize: "0.62rem", fontWeight: "700",
    background: (AVAIL_COLORS[avail] || COLORS.muted) + "22",
    color: AVAIL_COLORS[avail] || COLORS.muted,
    border: `1px solid ${(AVAIL_COLORS[avail] || COLORS.muted)}44`,
    fontFamily: "'Space Mono', monospace",
  }),
  metaRow: {
    display: "flex", gap: "14px", fontSize: "0.72rem", color: COLORS.muted,
    marginBottom: "6px", fontFamily: "'Space Mono', monospace", flexWrap: "wrap",
  },
  metaVal: { color: COLORS.text, fontWeight: "600" },
  reasonRow: {
    display: "flex", gap: "6px", flexWrap: "wrap", marginTop: "6px",
  },
  reason: {
    fontSize: "0.72rem", color: COLORS.muted,
    background: COLORS.panel, borderRadius: "4px", padding: "2px 7px",
  },
  scoreBar: {
    height: "3px", borderRadius: "2px", background: COLORS.border,
    marginTop: "8px", overflow: "hidden",
  },
  scoreBarFill: (score, color) => ({
    height: "100%", width: `${score * 100}%`,
    background: color, borderRadius: "2px",
    transition: "width 0.5s ease",
  }),
  emptyMsg: {
    padding: "30px 20px", textAlign: "center",
    color: COLORS.muted, fontSize: "0.82rem",
  },
};

export default function RecommendationPanel({
  userLat = 37.7749,
  userLon = -122.4194,
}) {
  const [connector,    setConnector]    = useState("Any");
  const [maxWait,      setMaxWait]      = useState(20);
  const [maxDist,      setMaxDist]      = useState(10);
  const [preferFree,   setPreferFree]   = useState(false);
  const [topK,         setTopK]         = useState(5);
  const [loading,      setLoading]      = useState(false);
  const [results,      setResults]      = useState([]);
  const [searched,     setSearched]     = useState(false);

  const handleSearch = useCallback(async () => {
    setLoading(true);
    setSearched(true);
    try {
      const res  = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_lat:             userLat,
          user_lon:             userLon,
          max_wait_minutes:     maxWait,
          max_distance_km:      maxDist,
          preferred_connector:  connector === "Any" ? "" : connector,
          prefer_free:          preferFree,
          top_k:                topK,
        }),
      });
      const json = await res.json();
      setResults(json.recommendations || []);
    } catch {
      // Demo fallback
      const rng  = s => { let x = Math.sin(s * 9999) * 9999; return x - Math.floor(x); };
      const avails = ["Available", "Available", "Moderate", "Busy", "Full"];
      const conns  = ["CCS,CHAdeMO", "Tesla", "J1772", "CCS", "CHAdeMO"];
      const names  = ["Downtown FastCharge", "SoMa Supercharger", "Market St Hub",
                       "Caltrain Plaza EV", "Embarcadero Point"];
      const demo = Array.from({ length: Math.min(topK, 5) }, (_, i) => ({
        rank: i + 1, station_id: i + 1,
        name:           names[i],
        distance_km:    +(rng(i * 3) * maxDist * 0.8 + 0.3).toFixed(1),
        wait_min:       Math.round(rng(i * 7) * maxWait * 0.9 + 2),
        n_connectors:   Math.round(rng(i * 11) * 10 + 2),
        availability:   avails[i],
        connector_types:conns[i],
        is_free:        rng(i * 13) > 0.6,
        score:          +(0.95 - i * 0.12 + rng(i) * 0.05).toFixed(3),
        reasons: [
          i === 0 ? "⚡ Available now" : null,
          rng(i * 17) < 0.4 ? "📍 Very close" : null,
          rng(i * 19) > 0.7 ? "🆓 Free charging" : null,
          connector !== "Any" && conns[i].includes(connector) ? `✅ ${connector} compatible` : null,
        ].filter(Boolean),
      }));
      setResults(demo);
    }
    setLoading(false);
  }, [userLat, userLon, connector, maxWait, maxDist, preferFree, topK]);

  return (
    <div style={css.wrapper}>
      <div style={css.header}>🏆 Station Recommendations</div>
      <div style={css.body}>

        {/* Filters */}
        <div style={css.filterGrid}>
          <div>
            <label style={css.label}>Connector Type</label>
            <select style={css.select} value={connector}
              onChange={e => setConnector(e.target.value)}>
              {CONNECTORS.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label style={css.label}>Max Wait (min)</label>
            <input style={css.input} type="number" min={5} max={90}
              value={maxWait} onChange={e => setMaxWait(+e.target.value)}/>
          </div>
          <div>
            <label style={css.label}>Max Distance (km)</label>
            <input style={css.input} type="number" min={1} max={50}
              value={maxDist} onChange={e => setMaxDist(+e.target.value)}/>
          </div>
          <div>
            <label style={css.label}>Top K Results</label>
            <select style={css.select} value={topK}
              onChange={e => setTopK(+e.target.value)}>
              {[3, 5, 10].map(k => <option key={k} value={k}>{k} stations</option>)}
            </select>
          </div>
        </div>

        <label style={css.checkRow}>
          <input type="checkbox" checked={preferFree}
            onChange={e => setPreferFree(e.target.checked)}
            style={{ accentColor: COLORS.green }}/>
          🆓 Prefer free charging stations
        </label>

        <button style={css.btn} onClick={handleSearch} disabled={loading}>
          {loading ? "🔍 Searching…" : "🔍 Find Best Stations"}
        </button>

        {/* Results */}
        {searched && results.length > 0 && (
          <>
            <div style={css.divider}/>
            {results.map((rec) => {
              const avail = rec.availability || "Moderate";
              const color = AVAIL_COLORS[avail] || COLORS.muted;
              return (
                <div key={rec.rank} style={css.recCard(rec.rank, avail)}>
                  <div style={css.recHeader}>
                    <div>
                      <span style={css.rankBadge}>#{rec.rank} </span>
                      <span style={css.recName}>{rec.name}</span>
                    </div>
                    <span style={css.statusBadge(avail)}>{avail}</span>
                  </div>

                  <div style={css.metaRow}>
                    <span>⏱️ <span style={css.metaVal}>{rec.wait_min} min</span></span>
                    <span>📍 <span style={css.metaVal}>{rec.distance_km} km</span></span>
                    <span>🔌 <span style={css.metaVal}>{rec.n_connectors}</span></span>
                    <span>{rec.connector_types}</span>
                    {rec.is_free && <span style={{ color: COLORS.green, fontWeight: "700" }}>🆓 Free</span>}
                  </div>

                  {rec.reasons?.length > 0 && (
                    <div style={css.reasonRow}>
                      {rec.reasons.map((r, i) => (
                        <span key={i} style={css.reason}>{r}</span>
                      ))}
                    </div>
                  )}

                  <div style={css.scoreBar}>
                    <div style={css.scoreBarFill(rec.score, color)}/>
                  </div>
                  <div style={{ fontSize: "0.62rem", color: COLORS.muted, textAlign: "right", marginTop: "3px",
                                 fontFamily: "'Space Mono', monospace" }}>
                    Score: {rec.score}
                  </div>
                </div>
              );
            })}
          </>
        )}

        {searched && results.length === 0 && !loading && (
          <div style={css.emptyMsg}>
            <div style={{ fontSize: "1.4rem", marginBottom: "8px" }}>🔍</div>
            No stations match your current filters.<br/>
            Try increasing max distance or wait time.
          </div>
        )}
      </div>
    </div>
  );
}
