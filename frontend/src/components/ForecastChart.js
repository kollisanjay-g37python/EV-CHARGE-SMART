// frontend/src/components/ForecastChart.js
// SVG bar chart of 12-24 hour wait-time forecast.
// No chart library dependency — pure SVG for minimal bundle size.

import React, { useMemo } from "react";

const TEAL = "#00d4aa";
const BLUE = "#3b82f6";
const ORANGE = "#f59e0b";

export default function ForecastChart({ data = [], currentHour }) {
  const maxWait = useMemo(() => Math.max(...data.map((d) => d.predicted_wait_min || 0), 1), [data]);

  if (!data.length) return <div className="chart-empty">No forecast data</div>;

  const BAR_W = 16;
  const GAP = 4;
  const H = 100;
  const W = data.length * (BAR_W + GAP);

  return (
    <div className="forecast-chart">
      <svg width="100%" viewBox={`0 0 ${W + 20} ${H + 30}`} preserveAspectRatio="xMidYMid meet">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75, 1].map((pct) => (
          <g key={pct}>
            <line
              x1="0" y1={H - H * pct}
              x2={W + 20} y2={H - H * pct}
              stroke="#1e2d45" strokeWidth="1"
            />
            <text x="0" y={H - H * pct - 2} fill="#334155" fontSize="7">
              {Math.round(maxWait * pct)}m
            </text>
          </g>
        ))}

        {data.map((point, i) => {
          const x = i * (BAR_W + GAP) + 10;
          const barH = Math.max(2, (point.predicted_wait_min / maxWait) * H);
          const y = H - barH;
          const isCurrent = point.hour === currentHour;
          const barColor = isCurrent ? ORANGE : point.predicted_wait_min > 30 ? "#ef4444" : point.predicted_wait_min > 15 ? ORANGE : TEAL;

          return (
            <g key={i}>
              {/* Bar */}
              <rect
                x={x} y={y} width={BAR_W} height={barH}
                fill={barColor}
                opacity={isCurrent ? 1 : 0.7}
                rx="2"
              />
              {/* Current hour indicator */}
              {isCurrent && (
                <text x={x + BAR_W / 2} y={y - 4} textAnchor="middle" fill={ORANGE} fontSize="6" fontWeight="700">
                  NOW
                </text>
              )}
              {/* X-axis label every 4 bars */}
              {i % 3 === 0 && (
                <text x={x + BAR_W / 2} y={H + 12} textAnchor="middle" fill="#64748b" fontSize="7">
                  {point.label}
                </text>
              )}
            </g>
          );
        })}

        {/* Baseline */}
        <line x1="10" y1={H} x2={W + 10} y2={H} stroke="#1e2d45" strokeWidth="1" />
      </svg>

      {/* Legend */}
      <div className="forecast-legend">
        <span style={{ color: TEAL }}>■ Normal</span>
        <span style={{ color: ORANGE }}>■ High</span>
        <span style={{ color: "#ef4444" }}>■ Critical</span>
        <span style={{ color: ORANGE }}>■ Current Hour</span>
      </div>
    </div>
  );
}
