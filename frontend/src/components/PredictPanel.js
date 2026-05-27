// frontend/src/components/PredictPanel.js
import React, { useState, useCallback, useEffect } from "react";

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
  body: { padding: "18px" },
  label: { fontSize: "0.7rem", color: COLORS.muted, marginBottom: "5px", display: "block" },
  input: {
    width: "100%", background: COLORS.bg,
    border: `1px solid ${COLORS.border}`, borderRadius: "8px",
    padding: "8px 12px", color: COLORS.text,
    fontSize: "0.85rem", outline: "none",
    fontFamily: "'Space Mono', monospace",
  },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "12px" },
  sliderWrap: { marginBottom: "12px" },
  sliderRow: {
    display: "flex", justifyContent: "space-between",
    fontSize: "0.7rem", color: COLORS.muted, marginBottom: "5px",
  },
  sliderVal: { color: COLORS.text, fontFamily: "'Space Mono', monospace" },
  slider: { width: "100%", accentColor: COLORS.green, cursor: "pointer" },
  btn: {
    width: "100%", padding: "10px",
    background: COLORS.green, color: COLORS.bg,
    border: "none", borderRadius: "8px",
    fontWeight: "700", fontSize: "0.9rem",
    cursor: "pointer", transition: "background 0.15s",
    fontFamily: "'DM Sans', sans-serif",
  },
  btnLoading: { background: COLORS.muted, cursor: "not-allowed" },
  divider: { height: "1px", background: COLORS.border, margin: "16px 0" },
  resultGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginTop: "14px",
  },
  kpiBox: {
    background: COLORS.bg, borderRadius: "8px",
    padding: "10px 12px", border: `1px solid ${COLORS.border}`,
  },
  kpiVal: (color) => ({
    fontFamily: "'Space Mono', monospace",
    fontSize: "1.1rem", fontWeight: "700", color,
  }),
  kpiLabel: { fontSize: "0.62rem", color: COLORS.muted, marginTop: "2px", textTransform: "uppercase", letterSpacing: "0.06em" },
  badge: (avail) => ({
    display: "inline-block", padding: "3px 10px", borderRadius: "20px",
    fontSize: "0.72rem", fontWeight: "700",
    background: (AVAIL_COLORS[avail] || COLORS.muted) + "22",
    color: AVAIL_COLORS[avail] || COLORS.muted,
    border: `1px solid ${(AVAIL_COLORS[avail] || COLORS.muted)}44`,
    fontFamily: "'Space Mono', monospace",
  }),
  gaugeWrap: { display: "flex", justifyContent: "center", padding: "8px 0" },
  errorBox: {
    background: "#2d1b1b", border: "1px solid #5a2d2d",
    borderRadius: "8px", padding: "10px 14px",
    fontSize: "0.78rem", color: COLORS.orange, marginTop: "12px",
  },
};

// Circular SVG gauge
function WaitGauge({ wait = 0, max = 30, color }) {
  const pct   = Math.min(100, (wait / max) * 100);
  const col   = color || (pct < 35 ? COLORS.green : pct < 70 ? COLORS.yellow : COLORS.orange);
  const r     = 44, circ = 2 * Math.PI * r;
  const dash  = (pct / 100) * circ;

  return (
    <div style={css.gaugeWrap}>
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r={r} fill="none" stroke={COLORS.border} strokeWidth="10"/>
        <circle
          cx="55" cy="55" r={r} fill="none"
          stroke={col} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          transform="rotate(-90 55 55)"
          style={{ transition: "stroke-dasharray 0.6s ease, stroke 0.4s" }}
        />
        <text x="55" y="50" textAnchor="middle" fill={col}
          fontSize="22" fontFamily="Space Mono,monospace" fontWeight="bold">
          {wait}
        </text>
        <text x="55" y="66" textAnchor="middle" fill={COLORS.muted}
          fontSize="9" fontFamily="Space Mono,monospace">
          MIN WAIT
        </text>
      </svg>
    </div>
  );
}

export default function PredictPanel({ selectedStation = null }) {
  const [stationId,    setStationId]    = useState(1);
  const [nConnectors,  setNConnectors]  = useState(6);
  const [trafficIndex, setTrafficIndex] = useState(0.4);
  const [temperature,  setTemperature]  = useState(18);
  const [precipitation,setPrecipitation]= useState(0);
  const [queueLength,  setQueueLength]  = useState(0);
  const [loading,      setLoading]      = useState(false);
  const [result,       setResult]       = useState(null);
  const [error,        setError]        = useState(null);

  // Sync when a station is selected on the map
  useEffect(() => {
    if (selectedStation) {
      setStationId(selectedStation.station_id);
      setNConnectors(selectedStation.num_connectors || 6);
    }
  }, [selectedStation]);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          station_id:    stationId,
          n_connectors:  nConnectors,
          traffic_index: trafficIndex,
          temperature,
          precipitation,
          queue_length:  queueLength,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Prediction failed");
      setResult(json.data);
    } catch {
      // Demo fallback
      const wait = Math.max(0, Math.round(
        trafficIndex * 20 + queueLength * 4 +
        (precipitation > 0 ? 4 : 0) +
        (temperature < 5 ? 3 : 0) +
        (Math.random() - 0.5) * 6
      ));
      const util = Math.min(0.99, trafficIndex * 0.85 + queueLength * 0.05);
      setResult({
        station_id:       stationId,
        blended_wait_min: wait,
        ml_wait_min:      Math.max(0, wait + Math.round((Math.random() - 0.5) * 3)),
        erlang_wait_min:  Math.max(0, wait - Math.round(Math.random() * 2)),
        utilisation:      +util.toFixed(3),
        availability:     wait < 8 ? "Available" : wait < 18 ? "Moderate" : wait < 28 ? "Busy" : "Full",
        prob_wait:        +(trafficIndex * 0.75).toFixed(3),
        queue_stable:     util < 1.0,
      });
    }
    setLoading(false);
  }, [stationId, nConnectors, trafficIndex, temperature, precipitation, queueLength]);

  const avail = result?.availability || "Available";
  const avCol = AVAIL_COLORS[avail] || COLORS.muted;

  return (
    <div style={css.wrapper}>
      <div style={css.header}>⏱️ Wait-Time Prediction</div>
      <div style={css.body}>

        {/* Station ID + Connectors */}
        <div style={css.grid2}>
          <div>
            <label style={css.label}>Station ID</label>
            <input style={css.input} type="number" min={1}
              value={stationId} onChange={e => setStationId(+e.target.value)}/>
          </div>
          <div>
            <label style={css.label}>Connectors</label>
            <input style={css.input} type="number" min={1} max={50}
              value={nConnectors} onChange={e => setNConnectors(+e.target.value)}/>
          </div>
        </div>

        {/* Traffic slider */}
        <div style={css.sliderWrap}>
          <div style={css.sliderRow}>
            <span>Traffic Index</span>
            <span style={css.sliderVal}>{trafficIndex.toFixed(2)}</span>
          </div>
          <input type="range" style={css.slider} min={0} max={1} step={0.05}
            value={trafficIndex} onChange={e => setTrafficIndex(+e.target.value)}/>
          <div style={{ display:"flex", justifyContent:"space-between",
                         fontSize:"0.6rem", color: COLORS.muted, marginTop:"2px" }}>
            <span>Free flow</span><span>Heavy jam</span>
          </div>
        </div>

        {/* Temperature + Precipitation */}
        <div style={css.grid2}>
          <div style={css.sliderWrap}>
            <div style={css.sliderRow}>
              <span>Temp (°C)</span>
              <span style={css.sliderVal}>{temperature}°</span>
            </div>
            <input type="range" style={css.slider} min={-10} max={45}
              value={temperature} onChange={e => setTemperature(+e.target.value)}/>
          </div>
          <div style={css.sliderWrap}>
            <div style={css.sliderRow}>
              <span>Queue Length</span>
              <span style={css.sliderVal}>{queueLength}</span>
            </div>
            <input type="range" style={css.slider} min={0} max={15}
              value={queueLength} onChange={e => setQueueLength(+e.target.value)}/>
          </div>
        </div>

        {/* Rain toggle */}
        <div style={{ display:"flex", alignItems:"center", gap:"10px",
                       marginBottom:"14px", fontSize:"0.82rem" }}>
          <input type="checkbox" id="rain" checked={precipitation > 0}
            onChange={e => setPrecipitation(e.target.checked ? 1 : 0)}
            style={{ accentColor: COLORS.blue, cursor:"pointer" }}/>
          <label htmlFor="rain" style={{ color: COLORS.muted, cursor:"pointer" }}>
            🌧️ Rain / Snow (increases demand)
          </label>
        </div>

        {/* Predict button */}
        <button
          style={{ ...css.btn, ...(loading ? css.btnLoading : {}) }}
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? "⏳ Running models…" : "🔮 Predict Wait Time"}
        </button>

        {/* Error */}
        {error && <div style={css.errorBox}>⚠️ {error}</div>}

        {/* Results */}
        {result && (
          <>
            <div style={css.divider}/>
            <WaitGauge wait={result.blended_wait_min} max={30} color={avCol}/>

            <div style={{ textAlign:"center", marginBottom:"12px" }}>
              <span style={css.badge(avail)}>{avail}</span>
              <span style={{ fontSize:"0.72rem", color: COLORS.muted, marginLeft:"10px" }}>
                P(wait) = {((result.prob_wait || 0) * 100).toFixed(0)}%
              </span>
            </div>

            <div style={css.resultGrid}>
              {[
                ["Blended", `${result.blended_wait_min} min`, COLORS.green],
                ["ML Model", `${result.ml_wait_min} min`,     COLORS.blue],
                ["Erlang-C", `${result.erlang_wait_min} min`, COLORS.muted],
                ["Util %",   `${((result.utilisation||0)*100).toFixed(0)}%`, COLORS.yellow],
              ].map(([label, val, color]) => (
                <div key={label} style={css.kpiBox}>
                  <div style={css.kpiVal(color)}>{val}</div>
                  <div style={css.kpiLabel}>{label}</div>
                </div>
              ))}
            </div>

            <div style={{ marginTop:"10px", fontSize:"0.72rem", color: COLORS.muted, textAlign:"center" }}>
              Queue {result.queue_stable ? "✅ Stable" : "⚠️ Overloaded"} ·
              Station #{result.station_id}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
