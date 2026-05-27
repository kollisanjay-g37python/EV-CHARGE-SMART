// frontend/src/components/QueueGauge.js
// SVG arc gauge displaying M/M/c queue utilisation (rho)
// and Erlang-C wait estimate for a station.

import React, { useState, useEffect } from "react";
import { fetchQueueAnalysis } from "../services/api";

const TEAL = "#00d4aa";
const ORANGE = "#f59e0b";
const RED = "#ef4444";

function arcPath(cx, cy, r, startDeg, endDeg) {
  const toRad = (d) => ((d - 90) * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startDeg));
  const y1 = cy + r * Math.sin(toRad(startDeg));
  const x2 = cx + r * Math.cos(toRad(endDeg));
  const y2 = cy + r * Math.sin(toRad(endDeg));
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

export default function QueueGauge({ station, API_BASE }) {
  const [queueData, setQueueData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!station?.station_id) return;
    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchQueueAnalysis(API_BASE, station.station_id, {
          arrival_rate: Math.max(0.5, (station.queue_size || 2) + 1),
          service_rate: 3.0,
        });
        setQueueData(data);
      } catch {
        // Fallback mock
        setQueueData({
          rho: 0.65,
          erlang_c: 0.42,
          avg_wait_min: 18.5,
          avg_queue_length: 2.3,
          utilization_pct: 65,
          system_stable: true,
          throughput_per_hr: 9.5,
          num_ports: station?.num_ports || 4,
        });
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [station?.station_id, API_BASE]);

  if (loading) return <div className="gauge-loading">Computing queue model...</div>;
  if (!queueData) return null;

  const { rho, avg_wait_min, avg_queue_length, utilization_pct, system_stable, erlang_c } = queueData;
  const clampedRho = Math.min(rho, 0.99);

  // SVG arc gauge params
  const cx = 90, cy = 90, r = 68;
  const startDeg = -150, endDeg = 150;
  const range = endDeg - startDeg;
  const fillDeg = startDeg + range * clampedRho;
  const gaugeColor = clampedRho < 0.6 ? TEAL : clampedRho < 0.85 ? ORANGE : RED;

  return (
    <div className="queue-gauge">
      {/* Station name */}
      <div className="gauge-title">{station?.name?.split("–")[0].trim()}</div>

      {/* SVG Arc Gauge */}
      <div className="gauge-svg-wrap">
        <svg width="180" height="130" viewBox="0 0 180 130">
          {/* Background arc */}
          <path
            d={arcPath(cx, cy, r, startDeg, endDeg)}
            fill="none" stroke="#1e2d45" strokeWidth="12" strokeLinecap="round"
          />
          {/* Fill arc */}
          {clampedRho > 0 && (
            <path
              d={arcPath(cx, cy, r, startDeg, fillDeg)}
              fill="none" stroke={gaugeColor} strokeWidth="12" strokeLinecap="round"
            />
          )}
          {/* Threshold marks */}
          {[0.6, 0.85, 1.0].map((t) => {
            const deg = startDeg + range * t;
            const rad = ((deg - 90) * Math.PI) / 180;
            const x1 = cx + (r - 8) * Math.cos(rad);
            const y1 = cy + (r - 8) * Math.sin(rad);
            const x2 = cx + (r + 8) * Math.cos(rad);
            const y2 = cy + (r + 8) * Math.sin(rad);
            return (
              <line key={t} x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={t === 0.6 ? TEAL : t === 0.85 ? ORANGE : RED}
                strokeWidth="2" />
            );
          })}
          {/* Centre text */}
          <text x={cx} y={cy - 8} textAnchor="middle" fill={gaugeColor}
            fontSize="22" fontWeight="800" fontFamily="'JetBrains Mono', monospace">
            {(clampedRho * 100).toFixed(0)}%
          </text>
          <text x={cx} y={cy + 10} textAnchor="middle" fill="#64748b" fontSize="10">
            Utilisation (ρ)
          </text>
          <text x={cx} y={cy + 26} textAnchor="middle"
            fill={system_stable ? TEAL : RED} fontSize="9" fontWeight="700">
            {system_stable ? "STABLE ✓" : "OVERLOADED ✗"}
          </text>
        </svg>
      </div>

      {/* Stats grid */}
      <div className="gauge-stats">
        {[
          { label: "Avg Wait", value: `${avg_wait_min?.toFixed(1)} min`, color: gaugeColor },
          { label: "Avg Queue", value: avg_queue_length?.toFixed(1), color: ORANGE },
          { label: "Erlang-C P(wait)", value: `${(erlang_c * 100).toFixed(0)}%`, color: "#3b82f6" },
          { label: "Throughput", value: `${queueData.throughput_per_hr?.toFixed(1)}/hr`, color: TEAL },
        ].map(({ label, value, color }) => (
          <div className="gauge-stat" key={label}>
            <div className="gauge-stat-val" style={{ color }}>{value}</div>
            <div className="gauge-stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div className="gauge-formula">
        M/M/c Erlang-C · c={queueData.num_ports} servers · λ/μ analysis
      </div>
    </div>
  );
}
