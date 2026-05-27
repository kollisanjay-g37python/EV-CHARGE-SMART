import React from "react";

const MetricCard = ({ title, value, unit, icon }) => {
  return (
    <div
      style={{
        background: "#ffffff",
        borderRadius: "12px",
        padding: "16px",
        margin: "10px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        minWidth: "180px",
      }}
    >
      <div style={{ fontSize: "14px", color: "#666" }}>
        {icon} {title}
      </div>

      <div
        style={{
          fontSize: "28px",
          fontWeight: "bold",
          marginTop: "10px",
        }}
      >
        {value} {unit}
      </div>
    </div>
  );
};

export default MetricCard;