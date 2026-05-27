// frontend/src/components/StationList.js
import React, { useState } from "react";

const COLORS = {
  bg: "#0d1117", panel: "#161b22", border: "#21262d",
  green: "#39d353", blue: "#58a6ff", orange: "#f78166",
  yellow: "#e3b341", text: "#c9d1d9", muted: "#8b949e",
};

const AVAIL_COLOR = {
  Available: "#39d353",
  Moderate:  "#e3b341",
  Busy:      "#f78166",
  Full:      "#da3633",
};

const css = {
  wrapper: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    borderRadius: "12px",
    overflow: "hidden",
  },
  header: {
    padding: "14px 18px",
    borderBottom: `1px solid ${COLORS.border}`,
    display: "flex", justifyContent: "space-between", alignItems: "center",
  },
  headerTitle: {
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.72rem", color: COLORS.green,
    letterSpacing: "0.1em", textTransform: "uppercase",
  },
  count: {
    fontSize: "0.72rem", color: COLORS.muted,
    fontFamily: "'Space Mono', monospace",
  },
  searchBar: {
    padding: "10px 14px",
    borderBottom: `1px solid ${COLORS.border}`,
  },
  searchInput: {
    width: "100%", background: COLORS.bg,
    border: `1px solid ${COLORS.border}`, borderRadius: "8px",
    padding: "7px 12px", color: COLORS.text,
    fontSize: "0.82rem", outline: "none",
    fontFamily: "'DM Sans', sans-serif",
  },
  filterRow: {
    display: "flex", gap: "6px", padding: "8px 14px",
    borderBottom: `1px solid ${COLORS.border}`,
    flexWrap: "wrap",
  },
  filterChip: (active, color) => ({
    padding: "3px 10px", borderRadius: "20px",
    fontSize: "0.68rem", fontWeight: "700",
    border: `1px solid ${active ? color : COLORS.border}`,
    background: active ? color + "22" : "transparent",
    color: active ? color : COLORS.muted,
    cursor: "pointer", transition: "all 0.15s",
    fontFamily: "'Space Mono', monospace",
  }),
  list: { maxHeight: "420px", overflowY: "auto" },
  row: (selected) => ({
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "12px 18px", cursor: "pointer",
    borderBottom: `1px solid ${COLORS.border}`,
    background: selected ? "rgba(57,211,83,0.04)" : "transparent",
    borderLeft: selected ? `3px solid ${COLORS.green}` : "3px solid transparent",
    transition: "background 0.15s",
  }),
  leftCol: {},
  stationName: {
    fontWeight: 600, fontSize: "0.88rem", color: COLORS.text,
    marginBottom: "3px",
  },
  metaRow: {
    display: "flex", gap: "12px",
    fontSize: "0.72rem", color: COLORS.muted,
    fontFamily: "'Space Mono', monospace",
  },
  rightCol: { textAlign: "right", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "5px" },
  waitTime: (color) => ({
    fontFamily: "'Space Mono', monospace",
    fontSize: "1rem", fontWeight: "700", color,
  }),
  badge: (avail) => ({
    display: "inline-block", padding: "2px 8px",
    borderRadius: "20px", fontSize: "0.62rem", fontWeight: "700",
    background: (AVAIL_COLOR[avail] || COLORS.muted) + "22",
    color: AVAIL_COLOR[avail] || COLORS.muted,
    border: `1px solid ${(AVAIL_COLOR[avail] || COLORS.muted)}44`,
    fontFamily: "'Space Mono', monospace",
  }),
  emptyState: {
    padding: "40px 20px", textAlign: "center", color: COLORS.muted,
    fontSize: "0.85rem",
  },
  sortRow: {
    display: "flex", gap: "6px", padding: "6px 14px 8px",
    alignItems: "center",
  },
  sortLabel: { fontSize: "0.68rem", color: COLORS.muted },
  sortBtn: (active) => ({
    padding: "2px 9px", borderRadius: "6px", fontSize: "0.68rem",
    border: `1px solid ${active ? COLORS.green : COLORS.border}`,
    background: active ? COLORS.green + "22" : "transparent",
    color: active ? COLORS.green : COLORS.muted,
    cursor: "pointer", fontFamily: "'Space Mono', monospace",
  }),
};

const FILTERS = ["All", "Available", "Moderate", "Busy", "Full"];
const SORTS   = ["wait", "distance", "name"];

export default function StationList({ stations = [], selectedId, onSelect }) {
  const [query,      setQuery]      = useState("");
  const [filterAvail,setFilterAvail]= useState("All");
  const [sortBy,     setSortBy]     = useState("wait");

  const filtered = stations
    .filter(s => filterAvail === "All" || s.availability === filterAvail)
    .filter(s => s.name?.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === "wait")     return (a.blended_wait_min ?? 99) - (b.blended_wait_min ?? 99);
      if (sortBy === "distance") return (a.distance_km ?? 99) - (b.distance_km ?? 99);
      if (sortBy === "name")     return (a.name || "").localeCompare(b.name || "");
      return 0;
    });

  return (
    <div style={css.wrapper}>
      {/* Header */}
      <div style={css.header}>
        <span style={css.headerTitle}>⚡ Nearby Stations</span>
        <span style={css.count}>{filtered.length} / {stations.length}</span>
      </div>

      {/* Search */}
      <div style={css.searchBar}>
        <input
          style={css.searchInput}
          placeholder="🔍 Search station name…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </div>

      {/* Availability filters */}
      <div style={css.filterRow}>
        {FILTERS.map(f => (
          <button
            key={f}
            style={css.filterChip(filterAvail === f, AVAIL_COLOR[f] || COLORS.blue)}
            onClick={() => setFilterAvail(f)}
          >
            {f === "All" ? "All" : f}
          </button>
        ))}
      </div>

      {/* Sort row */}
      <div style={css.sortRow}>
        <span style={css.sortLabel}>Sort:</span>
        {SORTS.map(s => (
          <button key={s} style={css.sortBtn(sortBy === s)} onClick={() => setSortBy(s)}>
            {s === "wait" ? "⏱ Wait" : s === "distance" ? "📍 Dist" : "🔤 Name"}
          </button>
        ))}
      </div>

      {/* List */}
      <div style={css.list}>
        {filtered.length === 0 ? (
          <div style={css.emptyState}>
            <div style={{ fontSize: "1.4rem", marginBottom: "8px" }}>🔍</div>
            No stations match your filters
          </div>
        ) : (
          filtered.map(station => {
            const avail    = station.availability || "Moderate";
            const color    = AVAIL_COLOR[avail] || COLORS.muted;
            const wait     = station.blended_wait_min ?? station.wait_min ?? "?";
            const selected = station.station_id === selectedId;

            return (
              <div
                key={station.station_id}
                style={css.row(selected)}
                onClick={() => onSelect?.(station)}
              >
                <div style={css.leftCol}>
                  <div style={css.stationName}>{station.name || `Station #${station.station_id}`}</div>
                  <div style={css.metaRow}>
                    {station.distance_km != null && (
                      <span>📍 {Number(station.distance_km).toFixed(1)} km</span>
                    )}
                    {station.num_connectors != null && (
                      <span>🔌 {station.num_connectors}</span>
                    )}
                    {station.connector_types && (
                      <span style={{ color: COLORS.muted }}>{station.connector_types.split(",")[0]}</span>
                    )}
                    {station.is_free && (
                      <span style={{ color: COLORS.green }}>🆓</span>
                    )}
                  </div>
                </div>

                <div style={css.rightCol}>
                  <div style={css.waitTime(color)}>
                    {typeof wait === "number" ? `${wait} min` : wait}
                  </div>
                  <span style={css.badge(avail)}>{avail}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
