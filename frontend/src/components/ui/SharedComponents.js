// frontend/src/components/ui/SharedComponents.js
import React from "react";

const C = {
  bg:     "#0d1117", panel:  "#161b22", card:   "#1c2330",
  border: "#21262d", green:  "#39d353", blue:   "#58a6ff",
  orange: "#f78166", yellow: "#e3b341", red:    "#da3633",
  text:   "#c9d1d9", muted:  "#8b949e",
};

// ── Availability colours ──────────────────────────────────────────────────────
export const AVAIL_COLOR = {
  Available: C.green, Moderate: C.yellow, Busy: C.orange, Full: C.red,
};
export const AVAIL_BADGE = {
  Available: "badge-available", Moderate: "badge-moderate",
  Busy:      "badge-busy",      Full:     "badge-full",
};

// ── Badge ─────────────────────────────────────────────────────────────────────
export function Badge({ label, variant }) {
  const cls = AVAIL_BADGE[variant] || AVAIL_BADGE[label] || "badge-moderate";
  return <span className={`badge ${cls}`}>{label}</span>;
}

// ── KPI Card ─────────────────────────────────────────────────────────────────
export function KpiCard({ icon, label, value, unit, color, delta, deltaPos }) {
  return (
    <div style={{
      background: C.panel, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: "16px 18px",
      transition: "border-color .2s",
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = C.green}
      onMouseLeave={e => e.currentTarget.style.borderColor = C.border}
    >
      {icon && <div style={{ fontSize: "1.2rem", marginBottom: 6 }}>{icon}</div>}
      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: "1.8rem", fontWeight: 700, lineHeight: 1,
        color: color || C.green,
        textShadow: `0 0 16px ${(color || C.green).replace(")", ",0.35)").replace("rgb", "rgba")}`,
      }}>
        {value}
        {unit && <span style={{ fontSize: "1rem", color: C.muted, marginLeft: 4 }}>{unit}</span>}
      </div>
      <div style={{ fontSize: "0.70rem", color: C.muted, textTransform: "uppercase",
                    letterSpacing: "0.08em", marginTop: 6 }}>{label}</div>
      {delta && (
        <div style={{ fontSize: "0.72rem", color: deltaPos ? C.green : C.orange,
                      marginTop: 4, fontFamily: "var(--font-mono)" }}>
          {delta}
        </div>
      )}
    </div>
  );
}

// ── Circular Gauge ────────────────────────────────────────────────────────────
export function CircleGauge({ value, max = 30, size = 100 }) {
  const pct   = Math.min(100, (value / max) * 100);
  const color = pct < 33 ? C.green : pct < 66 ? C.yellow : C.orange;
  const r     = (size / 2) - 8;
  const circ  = 2 * Math.PI * r;
  const dash  = (pct / 100) * circ;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r}
        fill="none" stroke={C.border} strokeWidth={8}/>
      <circle cx={size/2} cy={size/2} r={r}
        fill="none" stroke={color} strokeWidth={8}
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`}
        transform={`rotate(-90 ${size/2} ${size/2})`}
        style={{ transition: "stroke-dasharray .6s ease" }}/>
      <text x={size/2} y={size/2 - 4} textAnchor="middle"
        fill={color} fontSize={size * 0.18} fontFamily="var(--font-mono)"
        fontWeight="700">
        {value}
      </text>
      <text x={size/2} y={size/2 + 13} textAnchor="middle"
        fill={C.muted} fontSize={size * 0.08} fontFamily="var(--font-mono)">
        min wait
      </text>
    </svg>
  );
}

// ── Utilisation Bar ───────────────────────────────────────────────────────────
export function UtilBar({ value = 0, label }) {
  const pct   = Math.min(100, value * 100);
  const color = pct < 40 ? C.green : pct < 70 ? C.yellow : C.orange;
  return (
    <div>
      {label && (
        <div style={{ display:"flex", justifyContent:"space-between",
                      fontSize:"0.72rem", marginBottom:4 }}>
          <span style={{ color: C.muted }}>{label}</span>
          <span style={{ fontFamily:"var(--font-mono)", color }}>{pct.toFixed(0)}%</span>
        </div>
      )}
      <div style={{ height:5, background:C.border, borderRadius:3, overflow:"hidden" }}>
        <div style={{
          height:"100%", width:`${pct}%`, background:color,
          borderRadius:3, transition:"width .5s ease",
        }}/>
      </div>
    </div>
  );
}

// ── Loading Spinner ───────────────────────────────────────────────────────────
export function Spinner({ size = 18 }) {
  return (
    <div style={{
      width:size, height:size, borderRadius:"50%",
      border:`2px solid ${C.border}`, borderTopColor:C.green,
      animation:"spin .7s linear infinite",
      display:"inline-block",
    }}/>
  );
}

// ── Section Label ─────────────────────────────────────────────────────────────
export function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily:"var(--font-mono)", fontSize:"0.65rem",
      color:C.green, textTransform:"uppercase", letterSpacing:"0.12em",
      marginBottom:12, display:"flex", alignItems:"center", gap:10,
    }}>
      {children}
      <span style={{ flex:1, height:1, background:C.border }}/>
    </div>
  );
}

// ── Metric Row ────────────────────────────────────────────────────────────────
export function MetricRow({ label, value, color }) {
  return (
    <div className="metric-row">
      <span className="label">{label}</span>
      <span className="value" style={{ color: color || C.text }}>{value}</span>
    </div>
  );
}

// ── Empty State ───────────────────────────────────────────────────────────────
export function EmptyState({ icon = "🔍", title, subtitle }) {
  return (
    <div style={{ textAlign:"center", padding:"32px 20px", color:C.muted }}>
      <div style={{ fontSize:"2rem", marginBottom:10 }}>{icon}</div>
      {title    && <div style={{ fontWeight:700, color:C.text, marginBottom:6 }}>{title}</div>}
      {subtitle && <div style={{ fontSize:"0.85rem" }}>{subtitle}</div>}
    </div>
  );
}

// ── Tooltip wrapper ───────────────────────────────────────────────────────────
export function Tooltip({ text, children }) {
  return <span data-tooltip={text}>{children}</span>;
}
