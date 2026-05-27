// frontend/src/components/WaitTimeBadge.js
// Small coloured badge showing predicted wait time.

import React from "react";

const LEVELS = [
  { max: 5,  color: "#22c55e", bg: "#22c55e22", border: "#22c55e44", label: "GO NOW" },
  { max: 15, color: "#eab308", bg: "#eab30822", border: "#eab30844", label: "FAST" },
  { max: 30, color: "#f97316", bg: "#f9731622", border: "#f9731644", label: "MODERATE" },
  { max: Infinity, color: "#ef4444", bg: "#ef444422", border: "#ef444444", label: "LONG" },
];

export default function WaitTimeBadge({ wait, showLabel = false }) {
  const w = parseFloat(wait) || 0;
  const level = LEVELS.find((l) => w <= l.max) || LEVELS[LEVELS.length - 1];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        background: level.bg,
        color: level.color,
        border: `1px solid ${level.border}`,
        borderRadius: 6,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.04em",
        whiteSpace: "nowrap",
      }}
    >
      ⏱ {Math.round(w)} min{showLabel && ` · ${level.label}`}
    </span>
  );
}
