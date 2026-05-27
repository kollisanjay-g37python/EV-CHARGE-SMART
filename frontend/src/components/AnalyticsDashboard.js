// frontend/src/components/AnalyticsDashboard.js
import React, { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  PieChart, Pie, Cell,
} from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const COLORS = {
  bg: "#0d1117", panel: "#161b22", border: "#21262d",
  green: "#39d353", blue: "#58a6ff", orange: "#f78166",
  yellow: "#e3b341", text: "#c9d1d9", muted: "#8b949e",
};

const CSS = {
  wrapper: { display: "flex", flexDirection: "column", gap: "20px" },
  row:     { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" },
  card: {
    background: COLORS.panel, border: `1px solid ${COLORS.border}`,
    borderRadius: "12px", padding: "18px",
  },
  cardFull: {
    background: COLORS.panel, border: `1px solid ${COLORS.border}`,
    borderRadius: "12px", padding: "18px",
  },
  cardTitle: {
    fontFamily: "'Space Mono', monospace", fontSize: "0.68rem",
    color: COLORS.green, letterSpacing: "0.1em", textTransform: "uppercase",
    marginBottom: "14px",
  },
  metricsRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" },
  kpi: {
    background: COLORS.bg, borderRadius: "8px", padding: "14px",
    border: `1px solid ${COLORS.border}`,
  },
  kpiVal: (color) => ({
    fontFamily: "'Space Mono', monospace", fontSize: "1.6rem",
    fontWeight: "700", color, lineHeight: 1,
  }),
  kpiLabel: {
    fontSize: "0.65rem", color: COLORS.muted, marginTop: "5px",
    textTransform: "uppercase", letterSpacing: "0.06em",
  },
  heatmapGrid: { display: "grid", gap: "2px" },
  heatCell: (intensity) => ({
    height: "18px", borderRadius: "2px", cursor: "default",
    background: `rgba(${Math.round(intensity * 220)}, ${Math.round((1-intensity) * 180)}, 50, ${0.5 + intensity * 0.5})`,
    transition: "opacity 0.2s",
  }),
  heatLabel: {
    fontSize: "0.55rem", color: COLORS.muted,
    fontFamily: "'Space Mono', monospace", textAlign: "center",
  },
  tooltipStyle: {
    background: COLORS.card, border: `1px solid ${COLORS.border}`,
    borderRadius: "8px", fontSize: "0.75rem",
    fontFamily: "'Space Mono', monospace",
  },
};

const CHART_TOOLTIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ ...CSS.tooltipStyle, padding: "8px 12px" }}>
      <div style={{ color: COLORS.muted, marginBottom: "4px" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: "700" }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
};

// Generate synthetic heatmap data
function buildHeatmap() {
  const days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  const rng   = (s) => { let x = Math.sin(s * 9999) * 9999; return x - Math.floor(x); };
  return days.map((day, di) =>
    Array.from({ length: 24 }, (_, h) => {
      const peak    = (h >= 7 && h <= 9) || (h >= 17 && h <= 19);
      const weekend = di >= 5 ? 0.7 : 1.0;
      const base    = peak ? 20 : 8;
      return Math.round((base + rng(di * 100 + h) * 12) * weekend);
    })
  );
}

// Hourly bar data
const HOURLY_DATA = Array.from({ length: 24 }, (_, h) => {
  const rng = (s) => { let x = Math.sin(s * 7777) * 7777; return x - Math.floor(x); };
  const peak = (h >= 7 && h <= 9) || (h >= 17 && h <= 19);
  const base = peak ? 20 : 8;
  return {
    hour:    `${h}:00`,
    wait:    Math.round(base + rng(h * 3) * 10),
    sessions:Math.round((peak ? 40 : 15) + rng(h * 7) * 20),
  };
});

// Weekly trend
const WEEKLY_DATA = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((day, i) => {
  const rng = (s) => { let x = Math.sin(s * 6666) * 6666; return x - Math.floor(x); };
  const we  = i >= 5;
  return {
    day, wait: Math.round((we ? 10 : 13) + rng(i) * 6),
    sessions: Math.round((we ? 280 : 380) + rng(i * 3) * 60),
  };
});

// Availability pie
const PIE_DATA = [
  { name: "Available", value: 43, color: COLORS.green  },
  { name: "Moderate",  value: 32, color: COLORS.yellow },
  { name: "Busy",      value: 18, color: COLORS.orange },
  { name: "Full",      value:  7, color: "#da3633"     },
];

export default function AnalyticsDashboard() {
  const [metrics, setMetrics] = useState({});
  const heatmap = buildHeatmap();

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => setMetrics(d))
      .catch(() => setMetrics({
        RF:   { MAE: 2.526, RMSE: 3.507, R2: 0.6381 },
        LSTM: { MAE: 2.58,  RMSE: 3.588, R2: 0.6222 },
      }));
  }, []);

  const DAYS  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  const HOURS = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div style={CSS.wrapper}>

      {/* Model KPIs */}
      <div style={CSS.cardFull}>
        <div style={CSS.cardTitle}>📊 Model Performance</div>
        <div style={CSS.metricsRow}>
          {[
            ["RF MAE",   `${metrics.RF?.MAE  ?? "—"} min`, COLORS.green],
            ["RF RMSE",  `${metrics.RF?.RMSE ?? "—"} min`, COLORS.blue],
            ["RF R²",    `${metrics.RF?.R2   ?? "—"}`,     COLORS.yellow],
            ["LSTM MAE", `${metrics.LSTM?.MAE?? "—"} min`, COLORS.orange],
          ].map(([label, val, color]) => (
            <div key={label} style={CSS.kpi}>
              <div style={CSS.kpiVal(color)}>{val}</div>
              <div style={CSS.kpiLabel}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Hourly wait + sessions */}
      <div style={CSS.row}>
        <div style={CSS.card}>
          <div style={CSS.cardTitle}>⏱️ Avg Wait by Hour</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={HOURLY_DATA} margin={{ top:4, right:8, left:-20, bottom:0 }}>
              <CartesianGrid vertical={false} stroke={COLORS.border}/>
              <XAxis dataKey="hour" tick={{ fill:COLORS.muted, fontSize:8 }}
                interval={2} tickLine={false} axisLine={false}/>
              <YAxis tick={{ fill:COLORS.muted, fontSize:8 }} tickLine={false} axisLine={false}/>
              <Tooltip content={<CHART_TOOLTIP/>}/>
              <Bar dataKey="wait" name="Wait (min)" radius={[3,3,0,0]}
                fill={COLORS.green} opacity={0.85}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={CSS.card}>
          <div style={CSS.cardTitle}>📅 Weekly Trend</div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={WEEKLY_DATA} margin={{ top:4, right:8, left:-20, bottom:0 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="4 4"/>
              <XAxis dataKey="day" tick={{ fill:COLORS.muted, fontSize:9 }}
                tickLine={false} axisLine={false}/>
              <YAxis tick={{ fill:COLORS.muted, fontSize:8 }} tickLine={false} axisLine={false}/>
              <Tooltip content={<CHART_TOOLTIP/>}/>
              <Legend wrapperStyle={{ fontSize:"0.72rem", color:COLORS.muted }}/>
              <Line type="monotone" dataKey="wait" name="Avg Wait"
                stroke={COLORS.green} strokeWidth={2.5} dot={{ r:3, fill:COLORS.green }}/>
              <Line type="monotone" dataKey="sessions" name="Sessions"
                stroke={COLORS.blue} strokeWidth={2} dot={{ r:3, fill:COLORS.blue }}
                yAxisId={0} opacity={0.7}/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Heatmap + Pie */}
      <div style={CSS.row}>
        <div style={CSS.cardFull}>
          <div style={CSS.cardTitle}>🌡️ Wait Heatmap — Day × Hour</div>
          <div style={{ overflowX: "auto" }}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "36px repeat(24, 1fr)",
              gap: "2px", minWidth: "520px",
            }}>
              {/* Header */}
              <div/>
              {HOURS.map(h => (
                <div key={h} style={{ ...CSS.heatLabel, marginBottom: "3px" }}>
                  {h % 3 === 0 ? `${h}h` : ""}
                </div>
              ))}
              {/* Rows */}
              {heatmap.map((row, di) => (
                <React.Fragment key={di}>
                  <div style={{ ...CSS.heatLabel, display:"flex", alignItems:"center",
                                 fontSize:"0.6rem", paddingRight:"4px" }}>
                    {DAYS[di]}
                  </div>
                  {row.map((val, h) => (
                    <div key={h}
                      title={`${DAYS[di]} ${h}:00 — ${val} min`}
                      style={CSS.heatCell(Math.min(1, val / 30))}
                    />
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>
          <div style={{ display:"flex", gap:"16px", marginTop:"10px", fontSize:"0.68rem", color:COLORS.muted }}>
            <span style={{ display:"flex", alignItems:"center", gap:"5px" }}>
              <span style={{ width:12, height:12, background:COLORS.green, borderRadius:2, display:"inline-block" }}/>
              Low wait
            </span>
            <span style={{ display:"flex", alignItems:"center", gap:"5px" }}>
              <span style={{ width:12, height:12, background:COLORS.yellow, borderRadius:2, display:"inline-block" }}/>
              Moderate
            </span>
            <span style={{ display:"flex", alignItems:"center", gap:"5px" }}>
              <span style={{ width:12, height:12, background:COLORS.orange, borderRadius:2, display:"inline-block" }}/>
              High wait
            </span>
          </div>
        </div>

        <div style={CSS.card}>
          <div style={CSS.cardTitle}>🔋 Station Availability</div>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={PIE_DATA} cx="50%" cy="50%"
                innerRadius={50} outerRadius={80}
                dataKey="value" paddingAngle={2}
                label={({ name, value }) => `${name}: ${value}%`}
                labelLine={false}
              >
                {PIE_DATA.map((entry, i) => (
                  <Cell key={i} fill={entry.color}/>
                ))}
              </Pie>
              <Tooltip
                formatter={(val, name) => [`${val}%`, name]}
                contentStyle={{ background:COLORS.panel, border:`1px solid ${COLORS.border}`,
                                 borderRadius:"8px", fontSize:"0.75rem" }}
                labelStyle={{ color: COLORS.muted }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display:"flex", flexDirection:"column", gap:"6px", marginTop:"8px" }}>
            {PIE_DATA.map(d => (
              <div key={d.name} style={{ display:"flex", alignItems:"center", justifyContent:"space-between",
                                          fontSize:"0.75rem" }}>
                <div style={{ display:"flex", alignItems:"center", gap:"8px" }}>
                  <div style={{ width:10, height:10, borderRadius:"50%",
                                 background:d.color, flexShrink:0 }}/>
                  <span style={{ color: COLORS.muted }}>{d.name}</span>
                </div>
                <span style={{ fontFamily:"'Space Mono',monospace",
                                color: d.color, fontWeight:"700" }}>{d.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
