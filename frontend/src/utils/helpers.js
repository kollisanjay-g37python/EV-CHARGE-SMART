// frontend/src/utils/helpers.js
// Pure utility functions shared across all components.

// ─── Wait-time helpers ────────────────────────────────────────────────────────

export const WAIT_LEVELS = [
  { max: 5,        color: "#22c55e", label: "GO NOW",        bg: "#22c55e22", border: "#22c55e44" },
  { max: 15,       color: "#eab308", label: "GOOD TIME",     bg: "#eab30822", border: "#eab30844" },
  { max: 30,       color: "#f97316", label: "MODERATE WAIT", bg: "#f9731622", border: "#f9731644" },
  { max: Infinity, color: "#ef4444", label: "LONG WAIT",     bg: "#ef444422", border: "#ef444444" },
];

export function waitLevel(waitMin) {
  return WAIT_LEVELS.find((l) => waitMin <= l.max) || WAIT_LEVELS[WAIT_LEVELS.length - 1];
}

export function waitColor(waitMin) {
  return waitLevel(waitMin).color;
}

export function waitLabel(waitMin) {
  return waitLevel(waitMin).label;
}

export function formatWait(waitMin) {
  if (waitMin == null) return "— min";
  return `${Math.round(waitMin)} min`;
}

// ─── Utilisation helpers ──────────────────────────────────────────────────────

export function utilisationPct(numPorts, availablePorts) {
  if (!numPorts || numPorts === 0) return 0;
  return Math.round(((numPorts - availablePorts) / numPorts) * 100);
}

export function utilisationColor(pct) {
  if (pct < 60)  return "#22c55e";
  if (pct < 80)  return "#f59e0b";
  return "#ef4444";
}

// ─── Traffic helpers ──────────────────────────────────────────────────────────

export function trafficLabel(score) {
  if (score < 0.2)  return "Free Flow";
  if (score < 0.4)  return "Light";
  if (score < 0.6)  return "Moderate";
  if (score < 0.8)  return "Heavy";
  return "Gridlock";
}

export function trafficColor(score) {
  if (score < 0.33) return "#22c55e";
  if (score < 0.66) return "#f59e0b";
  return "#ef4444";
}

// ─── Geospatial helpers ───────────────────────────────────────────────────────

export function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function formatDistance(km) {
  if (km == null) return "—";
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
}

export function googleMapsUrl(lat, lng) {
  return `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}&travelmode=driving`;
}

// ─── Time helpers ─────────────────────────────────────────────────────────────

export function hourLabel(h) {
  if (h === 0)  return "12 AM";
  if (h === 12) return "12 PM";
  return h < 12 ? `${h} AM` : `${h - 12} PM`;
}

export function dayLabel(d) {
  return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d] || "—";
}

export function dayLabelShort(d) {
  return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] || "—";
}

export function isWeekend(dayOfWeek) {
  return dayOfWeek === 5 || dayOfWeek === 6; // Sat=5 Sun=6 (Mon=0 base)
}

export function isRushHour(hour) {
  return (hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 19);
}

export function formatTimestamp(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString();
}

// ─── Number helpers ───────────────────────────────────────────────────────────

export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

export function round(val, decimals = 1) {
  const factor = 10 ** decimals;
  return Math.round(val * factor) / factor;
}

export function pct(numerator, denominator, decimals = 0) {
  if (!denominator) return 0;
  return round((numerator / denominator) * 100, decimals);
}

// ─── Status helpers ───────────────────────────────────────────────────────────

export const STATUS_CONFIG = {
  Operational: { color: "#22c55e", bg: "#22c55e22", border: "#22c55e44", icon: "✅" },
  Partial:     { color: "#f59e0b", bg: "#f59e0b22", border: "#f59e0b44", icon: "⚠️" },
  Offline:     { color: "#ef4444", bg: "#ef444422", border: "#ef444444", icon: "❌" },
  Unknown:     { color: "#64748b", bg: "#64748b22", border: "#64748b44", icon: "❓" },
};

export function statusConfig(status) {
  return STATUS_CONFIG[status] || STATUS_CONFIG.Unknown;
}

// ─── Connector helpers ────────────────────────────────────────────────────────

export const CONNECTOR_ICONS = {
  CCS: "🔵", CHAdeMO: "🟡", "Type 2": "🟢", Tesla: "🔴",
  J1772: "🔵", Unknown: "⚪",
};

export function connectorIcon(type) {
  return CONNECTOR_ICONS[type] || "⚪";
}

// ─── Local storage helpers ────────────────────────────────────────────────────

export function lsGet(key, fallback = null) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch { return fallback; }
}

export function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

// ─── Debounce ─────────────────────────────────────────────────────────────────

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
