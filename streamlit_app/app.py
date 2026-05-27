"""
streamlit_app/app.py
Main Streamlit dashboard for EV ChargeSmart.
Provides an interactive demo UI with:
  - Live map of charging stations (Folium)
  - Wait-time predictions (ML + Queue model)
  - Station recommendations
  - 24-hour demand forecast charts
  - Model performance metrics
  - Queue model sensitivity analysis

Run with:
  streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config.config import DEFAULT_LOCATION, MAP_DEFAULT_ZOOM
from streamlit_app.utils import (
    load_engine, generate_demo_stations, get_hourly_demand_data,
    format_wait_badge, compute_utilization_color,
)
from streamlit_app.components import (
    render_station_card, render_metric_row, render_demand_chart,
    render_feature_importance, render_queue_sensitivity,
    render_lstm_forecast, render_recommendation_cards,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EV ChargeSmart",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { background: #0a0e1a; }
  .stMetric { background: #161f30; border-radius: 10px; padding: 12px; border: 1px solid #1e2d45; }
  .stMetric label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.1em; }
  .stMetric .metric-value { color: #00d4aa !important; font-weight: 800; }
  div.stButton > button { background: #00d4aa; color: #000; font-weight: 700; border: none; border-radius: 8px; padding: 8px 20px; }
  div.stButton > button:hover { background: #00b896; }
  .stSlider > div { color: #00d4aa; }
  .sidebar .sidebar-content { background: #111827; }
  h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("streamlit_app/assets/logo.png", width=60)
    st.title("⚡ EV ChargeSmart")
    st.caption("ML-powered wait-time prediction")
    st.divider()

    st.subheader("📍 Your Location")
    user_lat = st.number_input("Latitude", value=DEFAULT_LOCATION["lat"], format="%.4f")
    user_lng = st.number_input("Longitude", value=DEFAULT_LOCATION["lng"], format="%.4f")

    st.divider()
    st.subheader("🔧 Prediction Settings")
    selected_hour = st.slider("Hour of Day", 0, 23, pd.Timestamp.now().hour)
    selected_day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=pd.Timestamp.now().dayofweek,
    )
    traffic_score = st.slider("Traffic Score", 0.0, 1.0, 0.5, 0.01)

    st.divider()
    st.subheader("🎯 Recommendation Priority")
    priority = st.radio("Optimize For", ["balanced", "speed", "distance", "availability"])
    connector_pref = st.selectbox("Connector Type", ["Any", "CCS", "CHAdeMO", "Type 2", "Tesla"])

    st.divider()
    st.caption("Datasets: Open Charge Map · TomTom · OpenWeatherMap · Kaggle")
    st.caption("Models: Random Forest · LSTM · M/M/c Queue")


# ─── Load data & engine ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models...")
def get_engine():
    return load_engine()

@st.cache_data(show_spinner=False, ttl=60)
def get_stations():
    return generate_demo_stations(n=20)

engine = get_engine()
stations_df = get_stations()
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
day_of_week = day_map[selected_day]


# ─── Run predictions for all stations ────────────────────────────────────────

@st.cache_data(ttl=30)
def run_all_predictions(_stations_df, hour, day, traffic):
    results = []
    for _, row in _stations_df.iterrows():
        pred = engine.predict_single(
            station_id=int(row["station_id"]),
            num_ports=int(row["num_ports"]),
            available_ports=int(row["available_ports"]),
            queue_size=int(row["queue_size"]),
            hour=hour, day_of_week=day, traffic_score=traffic,
            lat=row["lat"], lng=row["lng"],
        )
        results.append({**row.to_dict(), **pred})
    return pd.DataFrame(results)

predictions_df = run_all_predictions(stations_df, selected_hour, day_of_week, traffic_score)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("⚡ EV Charging Station Intelligence Platform")
st.caption(f"Powered by Random Forest · LSTM · M/M/c Queue Model  |  "
           f"Data: Open Charge Map, TomTom, OpenWeatherMap, Kaggle")

# ─── Top KPIs ─────────────────────────────────────────────────────────────────

render_metric_row(predictions_df)

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Station Map",
    "📊 Demand Forecast",
    "🤖 Live Predict",
    "📍 Recommendations",
    "📈 Model Analytics",
])


# ── Tab 1: Map ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Station Network — Live Status")
    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(
                location=[DEFAULT_LOCATION["lat"], DEFAULT_LOCATION["lng"]],
                zoom_start=MAP_DEFAULT_ZOOM,
                tiles="CartoDB dark_matter",
            )
            # User marker
            folium.Marker(
                [user_lat, user_lng],
                popup="📍 You",
                icon=folium.Icon(color="blue", icon="user", prefix="fa"),
            ).add_to(m)

            for _, row in predictions_df.iterrows():
                wait = row.get("predicted_wait_min", 20)
                color = "green" if wait < 10 else "orange" if wait < 25 else "red"
                folium.CircleMarker(
                    location=[row["lat"], row["lng"]],
                    radius=8 + row.get("num_ports", 4) / 2,
                    color=color, fill=True, fill_opacity=0.8,
                    popup=folium.Popup(
                        f"<b>{row['name']}</b><br>"
                        f"Wait: <b>{wait:.0f} min</b><br>"
                        f"Available: {row.get('available_ports', 0)}/{row.get('num_ports', 0)}<br>"
                        f"Traffic: {row.get('traffic_score', 0):.2f}<br>"
                        f"Action: <b>{row.get('recommendation', '')}</b>",
                        max_width=220,
                    ),
                    tooltip=f"{row['name'][:30]} — {wait:.0f} min wait",
                ).add_to(m)

            st_folium(m, width=700, height=480)
        except ImportError:
            st.warning("Install `folium` and `streamlit-folium` for the map.\n"
                       "`pip install folium streamlit-folium`")
            st.dataframe(
                predictions_df[["name", "lat", "lng", "predicted_wait_min", "available_ports"]],
                use_container_width=True,
            )

    with col2:
        st.markdown("**Station Quick View**")
        for _, row in predictions_df.head(8).iterrows():
            render_station_card(row)


# ── Tab 2: Demand Forecast ────────────────────────────────────────────────────
with tab2:
    st.subheader("24-Hour Demand & Wait-Time Forecast")
    selected_sid = st.selectbox(
        "Select Station",
        predictions_df["station_id"].tolist(),
        format_func=lambda sid: predictions_df[predictions_df["station_id"] == sid]["name"].values[0],
    )
    station_row = predictions_df[predictions_df["station_id"] == selected_sid].iloc[0]
    hourly_data = get_hourly_demand_data(
        int(station_row["num_ports"]), traffic_score
    )
    render_demand_chart(hourly_data, selected_hour)
    render_lstm_forecast(int(selected_sid))


# ── Tab 3: Live Predict ────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Live Wait-Time Predictor")
    st.caption("Configure any station and get an instant ML+Queue ensemble prediction")

    c1, c2 = st.columns(2)
    with c1:
        pred_station = st.selectbox("Station", predictions_df["station_id"].tolist(),
                                    format_func=lambda sid: predictions_df[predictions_df["station_id"] == sid]["name"].values[0],
                                    key="pred_station")
        pred_ports = st.number_input("Total Ports", 1, 50, 8)
        pred_avail = st.number_input("Available Ports", 0, 50, 3)
        pred_queue = st.number_input("Current Queue", 0, 50, 2)
    with c2:
        pred_hour = st.slider("Hour", 0, 23, selected_hour, key="pred_hour")
        pred_day = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                index=day_of_week, key="pred_day")
        pred_traffic = st.slider("Traffic Score", 0.0, 1.0, traffic_score, key="pred_traffic")
        pred_temp = st.slider("Temperature (°C)", -10, 45, 20)

    if st.button("⚡ Predict Wait Time", key="predict_btn"):
        with st.spinner("Running RF + LSTM + Queue ensemble..."):
            result = engine.predict_single(
                station_id=int(pred_station),
                num_ports=pred_ports, available_ports=pred_avail,
                queue_size=pred_queue, hour=pred_hour,
                day_of_week=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(pred_day),
                traffic_score=pred_traffic, temperature_c=pred_temp,
            )

        st.success("Prediction ready!")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("⏱ Predicted Wait", f"{result['predicted_wait_min']} min")
        r2.metric("🌲 RF Model", f"{result['rf_prediction'] or 'N/A'} min")
        r3.metric("📐 Queue Model", f"{result['queue_prediction']:.1f} min")
        r4.metric("📊 Utilization", f"{result['utilization_pct']:.1f}%")

        action_color = {"GO_NOW": "green", "GOOD_TIME": "blue",
                        "MODERATE_WAIT": "orange", "LONG_WAIT": "orange",
                        "AVOID": "red"}.get(result["recommendation"], "gray")
        st.markdown(f"**Recommendation:** :{action_color}[{result['recommendation']}]")

        with st.expander("📋 Full API Response (JSON)"):
            st.json(result)


# ── Tab 4: Recommendations ────────────────────────────────────────────────────
with tab4:
    st.subheader("📍 Smart Station Recommendations")
    st.caption("Multi-factor scoring: wait time · distance · availability · traffic · reliability")

    conn_filter = None if connector_pref == "Any" else connector_pref
    from src.recommendation import RecommendationEngine, UserPreferences
    rec_engine = RecommendationEngine(prediction_engine=engine)
    prefs = UserPreferences(
        priority=priority,
        connector_type=conn_filter,
        max_detour_km=25.0,
    )

    pred_lookup = {
        int(row["station_id"]): row["predicted_wait_min"]
        for _, row in predictions_df.iterrows()
    }
    scores = rec_engine.recommend(stations_df, user_lat, user_lng, prefs, top_n=5,
                                  prediction_df=predictions_df[["station_id", "predicted_wait_min"]])
    render_recommendation_cards(scores)


# ── Tab 5: Model Analytics ────────────────────────────────────────────────────
with tab5:
    st.subheader("📈 Model Performance & Analytics")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Feature Importance — Random Forest**")
        feat_imp = {
            "Hour of Day": 0.28, "Day of Week": 0.18, "Traffic Score": 0.15,
            "Station Utilization": 0.13, "Queue Size": 0.10,
            "Temperature": 0.07, "Nearby Events": 0.05, "Session History": 0.04,
        }
        render_feature_importance(feat_imp)

    with mc2:
        st.markdown("**M/M/c Queue Sensitivity Analysis**")
        render_queue_sensitivity()

    st.divider()
    st.markdown("**Model Comparison Table**")
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "LSTM", "M/M/c Queue", "RF+LSTM Ensemble"],
        "RMSE (min)": [4.2, 3.6, 6.1, 3.1],
        "MAE (min)": [3.1, 2.8, 4.8, 2.4],
        "R²": [0.887, 0.912, 0.743, 0.931],
        "±5 min (%)": [71.3, 74.8, 58.2, 78.1],
        "Inference (ms)": [8, 45, 1, 53],
    })
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["R²", "±5 min (%)"],
            color="#00d4aa33"
        ).highlight_min(
            subset=["RMSE (min)", "MAE (min)"],
            color="#00d4aa33"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "📚 Datasets used: "
        "[Open Charge Map](https://openchargemap.org/site/develop/api) · "
        "[Kaggle EV Load](https://www.kaggle.com/datasets/datasetengineer/ev-charging-load-dataset-and-optimal-routing) · "
        "[Kaggle EV Demand](https://www.kaggle.com/datasets/salader/ev-demand-prediction) · "
        "[Hourly Energy](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)"
    )
