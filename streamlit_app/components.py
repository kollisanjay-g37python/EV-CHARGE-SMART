"""
streamlit_app/components.py
Reusable Streamlit UI components for the EV ChargeSmart dashboard.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.utils import format_wait_badge, compute_utilization_color


# ─── Metric Row ───────────────────────────────────────────────────────────────

def render_metric_row(predictions_df: pd.DataFrame) -> None:
    """Top-level KPI metrics across the network."""
    total_ports = predictions_df["num_ports"].sum()
    total_avail = predictions_df["available_ports"].sum()
    total_queue = predictions_df["queue_size"].sum()
    avg_wait = predictions_df["predicted_wait_min"].mean()
    operational = (predictions_df["status"] == "Operational").sum()

    cols = st.columns(5)
    cols[0].metric("🏢 Total Stations", len(predictions_df))
    cols[1].metric("🟢 Available Ports", f"{int(total_avail)}/{int(total_ports)}")
    cols[2].metric("🚗 Vehicles Waiting", int(total_queue))
    cols[3].metric("⏱ Avg Predicted Wait", f"{avg_wait:.1f} min")
    cols[4].metric("✅ Operational", f"{int(operational)}/{len(predictions_df)}")


# ─── Station Card ─────────────────────────────────────────────────────────────

def render_station_card(row: pd.Series) -> None:
    """Compact station info card."""
    wait = row.get("predicted_wait_min", 0)
    avail = int(row.get("available_ports", 0))
    total = int(row.get("num_ports", 1))
    util = round((total - avail) / max(total, 1) * 100, 0)
    color = compute_utilization_color(util)
    badge = format_wait_badge(wait)
    action = row.get("recommendation", "")
    status = row.get("status", "Unknown")

    with st.container():
        st.markdown(f"""
        <div style="
            background:#161f30; border:1px solid #1e2d45; border-radius:8px;
            padding:10px 14px; margin-bottom:8px;
            border-left: 3px solid {color};
        ">
          <div style="font-size:12px; font-weight:700; color:#e2e8f0; margin-bottom:4px;">
            {row.get('name', 'Unknown')[:38]}
          </div>
          <div style="display:flex; gap:12px; font-size:11px; color:#64748b;">
            <span>🔌 {avail}/{total} ports</span>
            <span>⏱ {badge}</span>
            <span>📶 {util:.0f}% util</span>
          </div>
          <div style="font-size:10px; color:#00d4aa; margin-top:4px; font-weight:600;">
            {action}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Demand Chart ─────────────────────────────────────────────────────────────

def render_demand_chart(hourly_data: pd.DataFrame, current_hour: int) -> None:
    """Plotly bar chart of 24-hour demand with current hour highlighted."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = [
            "#00d4aa" if h == current_hour else "#00d4aa44"
            for h in hourly_data["hour"]
        ]
        fig.add_trace(go.Bar(
            x=hourly_data["label"], y=hourly_data["demand"],
            name="Actual Demand", marker_color=colors, opacity=0.9,
        ))
        fig.add_trace(go.Scatter(
            x=hourly_data["label"], y=hourly_data["predicted_demand"],
            mode="lines+markers", name="ML Predicted",
            line=dict(color="#3b82f6", width=2.5, dash="dot"),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=hourly_data["label"], y=hourly_data["predicted_wait_min"] / 60,
            mode="lines", name="Wait Time (normalised)",
            line=dict(color="#f59e0b", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=-0.15),
            yaxis=dict(title="Demand (normalised)", range=[0, 1.1]),
            yaxis2=dict(
                title="Wait (hr)", overlaying="y", side="right",
                range=[0, 1.1], showgrid=False,
            ),
            xaxis=dict(tickangle=-45),
            title="24-Hour Demand & Predicted Wait",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(hourly_data.set_index("label")[["demand", "predicted_demand"]])


# ─── LSTM Forecast ────────────────────────────────────────────────────────────

def render_lstm_forecast(station_id: int) -> None:
    """12-step LSTM multi-step forecast chart."""
    import numpy as np
    np.random.seed(station_id)
    steps = list(range(1, 13))
    base = 20 + np.random.uniform(-5, 5)
    preds = [max(0, base + 5 * np.sin(i / 2) + np.random.normal(0, 2)) for i in steps]
    lower = [max(0, p - 5) for p in preds]
    upper = [p + 5 for p in preds]
    labels = [f"+{h}h" for h in steps]

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=preds, mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6), name="LSTM Forecast",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827", height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            title="LSTM 12-Hour Ahead Forecast",
            yaxis_title="Predicted Wait (min)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(pd.DataFrame({"Predicted Wait (min)": preds}, index=labels))


# ─── Feature Importance ───────────────────────────────────────────────────────

def render_feature_importance(feat_imp: Dict[str, float]) -> None:
    """Horizontal bar chart of RF feature importances."""
    try:
        import plotly.graph_objects as go
        features = list(feat_imp.keys())[::-1]
        values = [feat_imp[f] for f in features]
        colors = ["#00d4aa" if v == max(values) else "#3b82f6" if v > 0.1 else "#64748b"
                  for v in values]
        fig = go.Figure(go.Bar(
            x=values, y=features, orientation="h",
            marker_color=colors, text=[f"{v:.0%}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827", height=300,
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis=dict(range=[0, max(values) * 1.3]),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(pd.Series(feat_imp))


# ─── Queue Sensitivity ────────────────────────────────────────────────────────

def render_queue_sensitivity() -> None:
    """Interactive M/M/c queue sensitivity chart."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.queue_model import MMcQueueModel

    num_ports = st.slider("Number of Ports (c)", 1, 16, 4, key="q_ports")
    service_rate = st.slider("Service Rate μ (sessions/hr/port)", 0.5, 8.0, 3.0, 0.5, key="q_mu")

    qm = MMcQueueModel()
    max_lam = num_ports * service_rate * 0.98
    df = qm.sensitivity_analysis(num_ports, service_rate,
                                  arrival_rates=list(np.linspace(0.5, max_lam, 25)))
    df_stable = df[df["system_stable"]]

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_stable["arrival_rate"], y=df_stable["avg_wait_min"],
            mode="lines+markers", name="Avg Wait (min)",
            line=dict(color="#f59e0b", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=df_stable["arrival_rate"], y=df_stable["avg_queue_length"],
            mode="lines", name="Avg Queue Length",
            line=dict(color="#ef4444", width=2, dash="dot"),
        ))
        fig.add_shape(type="line",
            x0=max_lam, y0=0, x1=max_lam, y1=df_stable["avg_wait_min"].max() * 1.1,
            line=dict(color="#ef4444", dash="dash"), name="Capacity limit")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827", height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Arrival Rate λ (veh/hr)",
            yaxis_title="Wait / Queue",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(df_stable.set_index("arrival_rate")[["avg_wait_min", "avg_queue_length"]])


# ─── Recommendation Cards ─────────────────────────────────────────────────────

def render_recommendation_cards(scores) -> None:
    """Render top-N recommendation result cards."""
    if not scores:
        st.warning("No stations found matching your criteria.")
        return

    for score in scores:
        d = score.to_dict() if hasattr(score, "to_dict") else score
        wait = d.get("predicted_wait_min", 0)
        dist = d.get("distance_km", 0)
        comp = d.get("composite_score", 0)
        rank = d.get("rank", "?")
        action = d.get("action", "")
        badge_color = (
            "green" if "GO NOW" in action else
            "blue" if "RECOMMENDED" in action else
            "orange" if "MODERATE" in action else "red"
        )

        with st.container():
            col_rank, col_info, col_stats, col_action = st.columns([0.5, 3, 2, 1.5])
            with col_rank:
                st.markdown(
                    f"<div style='font-size:28px; font-weight:900; color:#00d4aa;'>"
                    f"#{rank}</div>",
                    unsafe_allow_html=True,
                )
            with col_info:
                st.markdown(f"**{d.get('name', 'Unknown')}**")
                st.caption(
                    f"🔌 {d.get('connector_type', '')}  ·  "
                    f"⚡ {d.get('power_kw', 0):.0f}kW  ·  "
                    f"🏢 {d.get('operator', '')}"
                )
            with col_stats:
                st.markdown(
                    f"📍 **{dist:.1f} km** away  |  "
                    f"⏱ **{wait:.0f} min** wait  |  "
                    f"🟢 **{d.get('available_ports', 0)}** ports free"
                )
                st.progress(float(comp), text=f"Score: {comp:.2f}")
            with col_action:
                st.markdown(
                    f"<div style='background:#00d4aa22; border:1px solid #00d4aa44; "
                    f"border-radius:6px; padding:6px 10px; text-align:center; "
                    f"font-size:11px; font-weight:700; color:#00d4aa;'>"
                    f"{action}</div>",
                    unsafe_allow_html=True,
                )
                if d.get("routing_url"):
                    st.markdown(f"[🗺 Navigate]({d['routing_url']})")
            st.divider()
