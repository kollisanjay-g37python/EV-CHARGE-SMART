// frontend/src/components/AlertBanner.js
import React, { useState } from "react";

const C = {
  panel: "#161b22", border: "#21262d",
  orange: "#f78166", yellow: "#e3b341",
  green: "#39d353",  text: "#c9d1d9", muted: "#8b949e",
};

/**
 * AlertBanner — dismissible toast notifications for wait-time alerts.
 * @param {{ alerts: Array, onDismiss: Function }} props
 */
export default function AlertBanner({ alerts = [], onDismiss }) {
  const [collapsed, setCollapsed] = useState(false);

  if (!alerts.length) return null;

  return (
    <div style={{
      position: "fixed", bottom: 24, right: 24,
      zIndex: 9000, maxWidth: 340, width: "100%",
    }}>
      {/* Header */}
      <div style={{
        background: C.panel, border: `1px solid ${C.orange}`,
        borderRadius: "10px 10px 0 0", padding: "8px 14px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        cursor: "pointer",
      }} onClick={() => setCollapsed(v => !v)}>
        <span style={{ color: C.orange, fontWeight: 700, fontSize: "0.82rem" }}>
          ⚠️ {alerts.length} Wait Alert{alerts.length > 1 ? "s" : ""}
        </span>
        <span style={{ color: C.muted, fontSize: "0.75rem" }}>
          {collapsed ? "▲ Show" : "▼ Hide"}
        </span>
      </div>

      {/* Alert list */}
      {!collapsed && (
        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderTop: "none", borderRadius: "0 0 10px 10px",
          maxHeight: 300, overflowY: "auto",
        }}>
          {alerts.slice(0, 5).map((alert, i) => (
            <AlertItem key={`${alert.station_id}-${i}`}
                       alert={alert}
                       onDismiss={() => onDismiss?.(alert.station_id)}/>
          ))}
          {alerts.length > 5 && (
            <div style={{ padding: "8px 14px", fontSize: "0.72rem",
                          color: C.muted, textAlign: "center",
                          borderTop: `1px solid ${C.border}` }}>
              +{alerts.length - 5} more alerts
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AlertItem({ alert, onDismiss }) {
  const waitColor = alert.wait_min >= 25 ? C.orange : C.yellow;
  return (
    <div style={{
      padding: "10px 14px",
      borderBottom: `1px solid ${C.border}`,
      display: "flex", justifyContent: "space-between", alignItems: "flex-start",
      gap: 10,
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600, fontSize: "0.82rem", color: C.text,
                      marginBottom: 2 }}>
          {alert.name || `Station #${alert.station_id}`}
        </div>
        <div style={{ fontSize: "0.74rem" }}>
          <span style={{ color: waitColor, fontFamily: "var(--font-mono)",
                          fontWeight: 700 }}>
            {alert.wait_min} min
          </span>
          <span style={{ color: C.muted, marginLeft: 8 }}>
            {alert.availability}
          </span>
        </div>
      </div>
      <button onClick={onDismiss} style={{
        background: "none", border: "none",
        color: C.muted, cursor: "pointer", fontSize: "1rem",
        flexShrink: 0, padding: 0,
      }}>✕</button>
    </div>
  );
}
