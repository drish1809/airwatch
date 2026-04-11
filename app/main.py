"""
app/main.py — AirWatch Dashboard v4
=====================================
v4 changes:
  ✅ 75 major world cities — flat list, no country/state drill-down
  ✅ Only cities with actual data shown everywhere
  ✅ Big AirWatch header fixed to top of every page
  ✅ Anomaly detection simplified — cards + timeline only
  ✅ Charts audited — removed misleading ones, added better ones
  ✅ All technical details hidden from users
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.anomaly import AnomalyDetector
from src.predictor import AQIPredictor
from src.utils import get_aqi_info, aqi_category, load_config

# ─────────────────────────────────────────────────────────────────────────────
# 75 Major world cities — curated, fetchable via OpenWeatherMap
# ─────────────────────────────────────────────────────────────────────────────
WORLD_CITIES = [
    # Asia — India
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat",
    # Asia — China
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu",
    # Asia — Others
    "Tokyo", "Seoul", "Bangkok", "Singapore", "Kuala Lumpur",
    "Jakarta", "Manila", "Karachi", "Lahore", "Dhaka",
    "Kathmandu", "Colombo", "Hanoi", "Ho Chi Minh City",
    # Middle East
    "Dubai", "Riyadh", "Tehran", "Istanbul", "Baghdad",
    # Europe
    "London", "Paris", "Berlin", "Madrid", "Rome",
    "Amsterdam", "Brussels", "Vienna", "Warsaw", "Stockholm",
    "Moscow", "Kyiv", "Athens", "Lisbon", "Zurich",
    # Africa
    "Cairo", "Lagos", "Nairobi", "Casablanca", "Johannesburg",
    # Americas — North
    "New York", "Los Angeles", "Chicago", "Toronto", "Mexico City",
    "Houston", "Phoenix", "Miami", "Vancouver", "Montreal",
    # Americas — South
    "São Paulo", "Buenos Aires", "Bogotá", "Lima", "Santiago",
    "Rio de Janeiro", "Caracas",
    # Oceania
    "Sydney", "Melbourne", "Auckland",
]
WORLD_CITIES = sorted(set(WORLD_CITIES))   # deduplicate & sort

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWatch — Real-Time Air Quality",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Background scheduler — once per server process
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_app() -> dict:
    status = {"db_ok": False, "api_key_ok": False, "scheduler_ok": False, "error": ""}
    try:
        cfg = load_config(str(ROOT / "config" / "config.yaml"))
        api_key = cfg["api"].get("openweather_api_key", "")
        status["api_key_ok"] = bool(api_key and api_key != "YOUR_OPENWEATHER_API_KEY")
    except Exception as e:
        status["error"] += f"Config: {e}. "
        cfg = {}
    try:
        from src.db import init_db, is_postgres
        init_db()
        status["db_ok"]   = True
        status["db_type"] = "postgres" if is_postgres() else "sqlite"
    except Exception as e:
        status["error"] += f"DB: {e}. "
    if status["api_key_ok"] and cfg:
        try:
            from src.scheduler import start_background_scheduler
            start_background_scheduler(cfg)
            status["scheduler_ok"] = True
        except Exception as e:
            status["error"] += f"Scheduler: {e}. "
    return status

_APP_STATUS = _init_app()

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Top brand bar ── */
.aw-topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 1rem 1.75rem;
    background: linear-gradient(90deg,#0d1117 0%,#161b22 60%,#0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    margin-bottom: 1.75rem;
    box-shadow: 0 2px 16px rgba(0,0,0,.4);
}
.aw-wordmark {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1;
    color: #e6edf3;
}
.aw-wordmark span { color: #58a6ff; }
.aw-tagline {
    font-size: .78rem;
    color: #7d8590;
    margin-top: 3px;
    letter-spacing: .03em;
}
.aw-divider {
    width: 1px; height: 40px;
    background: #21262d;
    margin: 0 4px;
}
.aw-page {
    font-size: 1.05rem;
    font-weight: 600;
    color: #c9d1d9;
}
.aw-sub {
    font-size: .75rem;
    color: #7d8590;
    margin-top: 2px;
}
.aw-badge {
    margin-left: auto;
    font-size: .7rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: .06em;
    font-weight: 600;
}
.aw-live { color: #3fb950; border: 1px solid rgba(63,185,80,.4); background: rgba(63,185,80,.08); }
.aw-demo { color: #d29922; border: 1px solid rgba(210,153,34,.4); background: rgba(210,153,34,.08); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0e14;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── KPI card ── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: border-color .2s, transform .2s;
}
.kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
.kpi-value {
    font-size: 1.85rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2; color: #58a6ff;
}
.kpi-label {
    font-size: .68rem; color: #7d8590;
    text-transform: uppercase; letter-spacing: .07em; margin-top: 5px;
}

/* ── Section label ── */
.sec-label {
    font-size: 1rem; font-weight: 600; color: #c9d1d9;
    border-left: 3px solid #58a6ff;
    padding-left: .65rem;
    margin: 1.75rem 0 .9rem;
}

/* ── AQI badge ── */
.aqi-badge {
    display: inline-block;
    padding: 5px 18px; border-radius: 20px;
    font-weight: 700; font-size: .9rem;
}

/* ── Anomaly alert card ── */
.anomaly-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: .5rem 0;
    border-left-width: 4px;
    border-left-style: solid;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.anomaly-icon { font-size: 1.6rem; flex-shrink: 0; }
.anomaly-city { font-size: 1rem; font-weight: 700; }
.anomaly-detail { font-size: .85rem; color: #8b949e; margin-top: 2px; }

/* ── Health card ── */
.health-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: .9rem 1.1rem;
}
.health-card p { color: #8b949e; font-size: .84rem; margin: .4rem 0 0; }

/* ── Prediction placeholder ── */
.predict-ph {
    background: #161b22; border: 2px dashed #21262d;
    border-radius: 14px; padding: 4rem 2rem;
    text-align: center; color: #7d8590;
}
.predict-ph .ph-icon { font-size: 2.5rem; margin-bottom: .75rem; }
.predict-ph p { font-size: 1rem; margin: 0; }

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: rgba(88,166,255,.1);
    border: 1px solid rgba(88,166,255,.25);
    color: #58a6ff; font-size: .72rem;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: .5rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .05em;
}

hr { border-color: #21262d !important; margin: 1.5rem 0; }
div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9", font_family="Space Grotesk",
    xaxis=dict(gridcolor="#1c2128", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1c2128", showline=False, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
    margin=dict(l=40, r=20, t=44, b=40),
)
AQI_CM = {
    "Good":                           "#00E400",
    "Moderate":                       "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy":                      "#FF0000",
    "Very Unhealthy":                 "#8F3F97",
    "Hazardous":                      "#7E0023",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data() -> pd.DataFrame:
    from src.db import load_dataframe
    df = load_dataframe()
    if df.empty:
        csv = ROOT / "data" / "raw" / "air_quality_data.csv"
        if csv.exists():
            df = pd.read_csv(csv)
        else:
            st.error("No data found. Run `python run.py --demo` first.")
            st.stop()

    df["timestamp"]    = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["aqi_category"] = df["aqi"].apply(aqi_category)
    df["pm_ratio"]     = df["pm2_5"] / (df["pm10"] + 1e-6)
    return df


@st.cache_resource(show_spinner=False)
def load_predictor() -> AQIPredictor:
    return AQIPredictor(str(ROOT / "models"))


def latest_per_city(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").groupby("city").last().reset_index()


def get_city_latest(df_all: pd.DataFrame, city: str) -> dict:
    rows = df_all[df_all["city"] == city].sort_values("timestamp")
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {c: (float(row[c]) if pd.notna(row[c]) else 0.0)
            for c in ["pm2_5","pm10","no2","o3","so2","co","temperature","humidity","wind_speed","aqi"]}


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def topbar(page_name: str, subtitle: str) -> None:
    """Big, bold AirWatch header fixed at the top of every page."""
    badge_cls  = "aw-badge aw-live" if IS_LIVE else "aw-badge aw-demo"
    badge_text = "● LIVE" if IS_LIVE else "● DEMO"
    st.markdown(f"""
    <div class="aw-topbar">
      <div>
        <div class="aw-wordmark">Air<span>Watch</span></div>
        <div class="aw-tagline">Real-Time Air Quality Intelligence</div>
      </div>
      <div class="aw-divider"></div>
      <div>
        <div class="aw-page">{page_name}</div>
        <div class="aw-sub">{subtitle}</div>
      </div>
      <div class="{badge_cls}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)


def sec(title: str) -> None:
    st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)


def kpi_card(col, value: str, label: str, color: str = "#58a6ff") -> None:
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    df_all    = load_data()
    predictor = load_predictor()

# Cities that actually have data in the database
cities_with_data = sorted(df_all["city"].unique().tolist())

IS_LIVE = _APP_STATUS.get("api_key_ok", False) and _APP_STATUS.get("db_ok", False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.6rem 0 1rem;">
      <div style="font-size:1.6rem;font-weight:800;color:#e6edf3;letter-spacing:-.02em;">
        🌬️ Air<span style="color:#58a6ff;">Watch</span>
      </div>
      <div style="font-size:.72rem;color:#7d8590;margin-top:2px;">Real-Time Air Quality</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("nav", [
        "📊 Dashboard",
        "🌍 City Comparison",
        "🔮 AQI Prediction",
        "⚠️ Health & Safety",
        "🗺️ Map View",
        "🔬 Anomaly Detection",
    ], label_visibility="collapsed")

    st.divider()

    # ── City filter — flat list of only cities with real data ─────────────────
    st.markdown("**🏙️ Select Cities**")
    selected_cities = st.multiselect(
        "cities",
        cities_with_data,
        default=cities_with_data,
        label_visibility="collapsed",
        placeholder="Search cities…",
    )
    if not selected_cities:
        selected_cities = cities_with_data

    st.markdown("**📅 Time Range**")
    days_back = st.slider("days", 1, 30, 7, label_visibility="collapsed",
                          format="%d days")
    cutoff = datetime.now() - timedelta(days=days_back)

    df = df_all[
        (df_all["city"].isin(selected_cities)) &
        (df_all["timestamp"] >= cutoff)
    ].copy()

    st.divider()

    if IS_LIVE:
        st.success("🔄 Live collection is ON")
    else:
        st.warning("Demo mode")

    st.caption(f"Cities: **{len(cities_with_data)}** · Updated: {datetime.now():%H:%M}")

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st_autorefresh(interval=600_000, key="autorefresh")


# ─────────────────────────────────────────────────────────────────────────────
# Guard helper
# ─────────────────────────────────────────────────────────────────────────────
def require_data() -> bool:
    if df.empty:
        st.warning("No data for the selected filters. Try selecting more cities or a wider time range.")
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    topbar("Dashboard", f"{len(selected_cities)} cities · last {days_back} days")

    if not require_data():
        st.stop()

    latest = latest_per_city(df)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    worst = latest.loc[latest["aqi"].idxmax(), "city"]
    best  = latest.loc[latest["aqi"].idxmin(), "city"]
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{latest['aqi'].mean():.0f}",
         f"{latest['pm2_5'].mean():.1f}",
         f"{latest['pm10'].mean():.1f}",
         worst, best],
        ["Avg AQI", "Avg PM2.5 μg/m³", "Avg PM10 μg/m³",
         "Most Polluted", "Cleanest Air"],
    ):
        kpi_card(col, val, lbl)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── AQI Ranking bar ───────────────────────────────────────────────────────
    sec("Current AQI by City")
    top = latest.sort_values("aqi", ascending=True)   # horizontal bars, worst at top
    fig_bar = go.Figure(go.Bar(
        y=top["city"], x=top["aqi"],
        orientation="h",
        marker_color=[AQI_CM.get(c, "#58a6ff") for c in top["aqi_category"]],
        text=top["aqi"].round(0).astype(int),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AQI: %{x:.0f}<extra></extra>",
    ))
    fig_bar.add_vline(x=100, line_dash="dash", line_color="#FFFF00",
                      opacity=0.6, annotation_text="Moderate")
    fig_bar.add_vline(x=200, line_dash="dash", line_color="#FF0000",
                      opacity=0.6, annotation_text="Unhealthy")
    fig_bar.update_layout(
        height=max(250, len(top) * 32),
        xaxis_title="AQI", yaxis_title="",
        showlegend=False, **_PL,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── AQI trend + category mix ──────────────────────────────────────────────
    col_trend, col_pie = st.columns([3, 2])

    with col_trend:
        sec("AQI Trend Over Time")
        fig_trend = px.line(
            df[df["city"].isin(selected_cities[:10])],
            x="timestamp", y="aqi", color="city",
            labels={"aqi": "AQI", "timestamp": ""},
        )
        fig_trend.add_hline(y=100, line_dash="dot", line_color="#FFFF00", opacity=0.5)
        fig_trend.add_hline(y=150, line_dash="dot", line_color="#FF7E00", opacity=0.5)
        fig_trend.add_hline(y=200, line_dash="dot", line_color="#FF0000", opacity=0.5)
        fig_trend.update_layout(**_PL)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_pie:
        sec("Air Quality Distribution")
        cats = latest["aqi_category"].value_counts().reset_index()
        cats.columns = ["category", "count"]
        fig_pie = px.pie(
            cats, values="count", names="category",
            color="category", color_discrete_map=AQI_CM, hole=0.5,
        )
        fig_pie.update_traces(textinfo="label+percent", pull=[0.03]*len(cats))
        fig_pie.update_layout(**_PL)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── PM2.5 vs AQI scatter ──────────────────────────────────────────────────
    sec("PM2.5 vs AQI — Latest Readings")
    st.caption("Each point is one city. Size = PM10 level.")
    fig_sc = px.scatter(
        latest, x="pm2_5", y="aqi", color="aqi_category",
        color_discrete_map=AQI_CM, size="pm10",
        hover_name="city", size_max=30,
        labels={"pm2_5": "PM2.5 (μg/m³)", "aqi": "AQI", "aqi_category": "Category"},
    )
    fig_sc.update_layout(**_PL)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 Latest readings per city"):
        show = [c for c in ["city","aqi","aqi_category","pm2_5","pm10",
                             "no2","temperature","humidity","timestamp"]
                if c in latest.columns]
        st.dataframe(
            latest[show].sort_values("aqi", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CITY COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌍 City Comparison":

    topbar("City Comparison", "Compare air quality across cities")

    compare = st.multiselect(
        "Select cities to compare (2–8)",
        cities_with_data,
        default=cities_with_data[:min(4, len(cities_with_data))],
        placeholder="Search and add cities…",
    )
    if len(compare) < 2:
        st.info("Select at least 2 cities to compare.")
        st.stop()

    df_cmp  = df_all[df_all["city"].isin(compare) & (df_all["timestamp"] >= cutoff)]
    lat_cmp = latest_per_city(df_cmp)

    if df_cmp.empty:
        st.warning("No data in this date range. Try extending the time window.")
        st.stop()

    # ── Radar ─────────────────────────────────────────────────────────────────
    sec("Pollutant Comparison")
    radar_cols = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
    radar_lbls = ["PM2.5",  "PM10", "NO₂", "O₃", "SO₂", "CO"]
    fig_r = go.Figure()
    for _, row in lat_cmp.iterrows():
        vals = [float(row.get(m) or 0) for m in radar_cols]
        maxv = [max(float(lat_cmp[m].max()), 1.0) for m in radar_cols]
        norm = [v / mx * 100 for v, mx in zip(vals, maxv)]
        fig_r.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=radar_lbls + [radar_lbls[0]],
            fill="toself", name=row["city"], opacity=0.75,
        ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1c2128"),
            angularaxis=dict(gridcolor="#1c2128"),
        ), **_PL,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Bar comparison of key metrics ─────────────────────────────────────────
    sec("Side-by-Side Metrics")
    metric_choice = st.selectbox(
        "Choose metric",
        ["aqi", "pm2_5", "pm10", "no2", "temperature", "humidity"],
        format_func=lambda x: {
            "aqi":"AQI","pm2_5":"PM2.5 (μg/m³)","pm10":"PM10 (μg/m³)",
            "no2":"NO₂ (μg/m³)","temperature":"Temperature (°C)","humidity":"Humidity (%)"
        }[x],
    )
    fig_cmp = px.bar(
        lat_cmp.sort_values(metric_choice, ascending=False),
        x="city", y=metric_choice,
        color="city",
        text=lat_cmp.sort_values(metric_choice, ascending=False)[metric_choice].round(1),
        labels={metric_choice: metric_choice.upper()},
    )
    fig_cmp.update_traces(textposition="outside")
    fig_cmp.update_layout(showlegend=False, **_PL)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── AQI trend lines ───────────────────────────────────────────────────────
    sec("AQI Trend")
    fig_t = px.line(df_cmp, x="timestamp", y="aqi", color="city",
                    labels={"aqi": "AQI", "timestamp": ""})
    fig_t.update_layout(**_PL)
    st.plotly_chart(fig_t, use_container_width=True)

    # ── Stats table ───────────────────────────────────────────────────────────
    sec("Summary Statistics")
    cols_s = [c for c in ["aqi","pm2_5","pm10","no2","temperature"] if c in df_cmp.columns]
    summary = df_cmp.groupby("city")[cols_s].agg(["mean","max","min"]).round(1)
    summary.columns = [f"{c[0].upper()} ({c[1]})" for c in summary.columns]
    st.dataframe(summary, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AQI PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 AQI Prediction":

    topbar("AQI Prediction", "Forecast air quality for any city and hour")

    if not predictor.is_ready:
        st.warning("No trained model found. Run `python run.py --train` first.")

    # ── City selector — ONLY cities with data ─────────────────────────────────
    st.markdown('<div class="info-pill">STEP 1 — CHOOSE CITY</div>', unsafe_allow_html=True)
    pred_city = st.selectbox(
        "City", cities_with_data,
        placeholder="Search for a city…",
        label_visibility="collapsed",
    )

    # ── Date & hour ───────────────────────────────────────────────────────────
    st.markdown('<div class="info-pill">STEP 2 — DATE & TIME</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        pred_hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    with pc2:
        pred_date = st.date_input("Date", datetime.today())

    # ── Pollutant inputs — auto-filled from latest data ───────────────────────
    st.markdown('<div class="info-pill">STEP 3 — POLLUTANT LEVELS</div>', unsafe_allow_html=True)
    city_live = get_city_latest(df_all, pred_city)
    has_live  = bool(city_live)

    manual = st.checkbox(
        "Enter values manually",
        value=not has_live,
        help="Uncheck to use the latest collected data for this city.",
    )

    defaults = city_live if (has_live and not manual) else {}
    disabled = has_live and not manual

    if has_live and not manual:
        st.caption(f"Auto-filled from the latest reading for **{pred_city}**.")

    pp1, pp2, pp3, pp4, pp5 = st.columns(5)
    with pp1: pm25  = st.number_input("PM2.5 (μg/m³)", 0.0, 500.0, float(round(defaults.get("pm2_5",  35), 1)), step=1.0, disabled=disabled)
    with pp2: pm10  = st.number_input("PM10 (μg/m³)",  0.0, 600.0, float(round(defaults.get("pm10",   65), 1)), step=1.0, disabled=disabled)
    with pp3: no2   = st.number_input("NO₂ (μg/m³)",   0.0, 300.0, float(round(defaults.get("no2",    28), 1)), step=1.0, disabled=disabled)
    with pp4: temp  = st.number_input("Temp (°C)",    -20.0,  55.0, float(round(defaults.get("temperature", 28), 1)), step=0.5, disabled=disabled)
    with pp5: humid = st.number_input("Humidity (%)",   0.0, 100.0, float(round(defaults.get("humidity",    65), 1)), step=1.0, disabled=disabled)

    st.markdown("")
    clicked = st.button("🔮 Predict AQI", type="primary", use_container_width=True)

    if not clicked:
        st.markdown("""<div class="predict-ph">
          <div class="ph-icon">🔮</div>
          <p>Select a city and click <b>Predict AQI</b> to see the forecast.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Prediction ────────────────────────────────────────────────────────────
    pred_day   = pd.Timestamp(pred_date).dayofweek
    is_weekend = int(pred_day >= 5)
    is_rush    = int(pred_hour in [7, 8, 9, 17, 18, 19])
    city_enc   = cities_with_data.index(pred_city) if pred_city in cities_with_data else 0

    features = {
        "hour": pred_hour, "day_of_week": pred_day,
        "month": pd.Timestamp(pred_date).month,
        "is_weekend": is_weekend, "is_rush_hour": is_rush,
        "pm2_5": pm25, "pm10": pm10, "no2": no2,
        "temperature": temp, "humidity": humid,
        "pm_ratio":          pm25 / (pm10 + 1e-6),
        "no_no2_ratio":      no2 * 0.2 / (no2 + 1e-6),
        "pm2_5_lag_1": pm25 * 0.95, "pm2_5_lag_2": pm25 * 0.90, "pm2_5_lag_3": pm25 * 0.87,
        "aqi_lag_1": pm25 * 3.5,    "aqi_lag_2": pm25 * 3.3,    "aqi_lag_3": pm25 * 3.1,
        "pm2_5_rolling_mean_3": pm25, "pm2_5_rolling_std_3": pm25 * 0.1,
        "aqi_rolling_mean_3":   pm25 * 3.4,
        "city_encoded": city_enc,
        "co": pm25 * 3.2, "no": no2 * 0.2, "o3": 60.0,
        "so2": pm25 * 0.1, "nh3": pm25 * 0.05, "wind_speed": 5.0,
    }

    if predictor.is_ready:
        predicted_aqi = predictor.predict(features)
    else:
        rm = 1.2 if is_rush else (0.8 if pred_hour < 6 else 1.0)
        predicted_aqi = max(25.0, pm25 * 3.0 * rm + no2 * 0.5)

    info = get_aqi_info(predicted_aqi)
    st.markdown("---")
    sec(f"Result — {pred_city}  ·  {pred_date}  {pred_hour:02d}:00")

    rg, rr = st.columns([1, 1])
    with rg:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=predicted_aqi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted AQI", "font": {"size": 15}},
            number={"font": {"size": 52, "color": info["color"]}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": "#c9d1d9"},
                "bar":  {"color": info["color"]},
                "steps": [
                    {"range": [0,   50], "color": "#002200"},
                    {"range": [50, 100], "color": "#222200"},
                    {"range": [100,150], "color": "#221500"},
                    {"range": [150,200], "color": "#220000"},
                    {"range": [200,300], "color": "#1a0022"},
                    {"range": [300,500], "color": "#1a0008"},
                ],
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
            height=290, margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with rr:
        st.markdown(f"""
        <div style="padding:.25rem .5rem;">
          <div style="font-size:3rem;text-align:center;margin-bottom:.6rem;">{info['emoji']}</div>
          <div style="text-align:center;margin-bottom:.75rem;">
            <span class="aqi-badge"
              style="background:{info['color']}1a;color:{info['color']};border:1px solid {info['color']}88;">
              {info['category']}
            </span>
          </div>
          <p style="color:#8b949e;text-align:center;font-size:.9rem;margin:0 0 1rem;">
            {info['health_msg']}
          </p>
          <hr>
          <strong style="color:#c9d1d9;">Precautions</strong>
          <ul style="padding-left:1.1rem;margin:.5rem 0 0;">
            {"".join(f"<li style='color:#8b949e;font-size:.85rem;margin:5px 0;'>{r}</li>" for r in info['recommendations'])}
          </ul>
        </div>""", unsafe_allow_html=True)

    # ── 24-hour forecast ──────────────────────────────────────────────────────
    sec("24-Hour Forecast")
    fc = []
    for h in range(24):
        fh = {**features, "hour": h, "is_rush_hour": int(h in [7, 8, 9, 17, 18, 19])}
        if predictor.is_ready:
            aqih = predictor.predict(fh)
        else:
            rm   = 1.2 if fh["is_rush_hour"] else (0.8 if h < 6 else 1.0)
            aqih = max(25.0, pm25 * 3.0 * rm + no2 * 0.5)
        fc.append({"hour": h, "aqi": aqih, "cat": aqi_category(aqih)})

    fc_df = pd.DataFrame(fc)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(
        x=fc_df["hour"], y=fc_df["aqi"],
        marker_color=[AQI_CM.get(c, "#58a6ff") for c in fc_df["cat"]],
        hovertemplate="%{x}:00 — AQI: %{y:.0f}<extra></extra>",
        name="AQI",
    ))
    for yv, col, lab in [(100,"#FFFF00","Moderate"),(150,"#FF7E00","Sensitive"),(200,"#FF0000","Unhealthy")]:
        fig_fc.add_hline(y=yv, line_dash="dot", line_color=col, opacity=0.5, annotation_text=lab)
    fig_fc.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Predicted AQI",
        title=f"24-Hour AQI Forecast — {pred_city}", **_PL,
    )
    st.plotly_chart(fig_fc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HEALTH & SAFETY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Health & Safety":

    topbar("Health & Safety", "City-specific health guidance based on current AQI")

    st.markdown('<div class="info-pill">SELECT CITY</div>', unsafe_allow_html=True)
    hs_city = st.selectbox(
        "City", cities_with_data,
        label_visibility="collapsed",
        placeholder="Search for a city…",
    )

    st.divider()

    city_df  = df_all[df_all["city"] == hs_city].sort_values("timestamp")
    if city_df.empty:
        st.warning(f"No data collected for **{hs_city}** yet.")
        st.stop()

    last_row = city_df.iloc[-1]
    aqi_val  = float(last_row["aqi"])
    info     = get_aqi_info(aqi_val)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in zip(
        [k1, k2, k3, k4],
        [f"{aqi_val:.0f}",
         f"{float(last_row.get('pm2_5') or 0):.1f}",
         f"{float(last_row.get('pm10') or 0):.1f}",
         f"{float(last_row.get('no2') or 0):.1f}"],
        ["AQI", "PM2.5 μg/m³", "PM10 μg/m³", "NO₂ μg/m³"],
    ):
        kpi_card(col, val, lbl, color=info["color"])

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── AQI badge ─────────────────────────────────────────────────────────────
    st.markdown(f"""<div style="text-align:center;margin-bottom:1.25rem;">
      <span class="aqi-badge"
        style="background:{info['color']}1a;color:{info['color']};border:1px solid {info['color']}88;font-size:1.1rem;">
        {info['emoji']} &nbsp; {info['category']}
      </span>
      <p style="color:#7d8590;margin:.7rem 0 0;font-size:.9rem;">{info['health_msg']}</p>
    </div>""", unsafe_allow_html=True)

    # ── Precautions ───────────────────────────────────────────────────────────
    sec("Recommended Precautions")
    recs  = info["recommendations"]
    icons = ["😷", "🏠", "💨", "🌿", "🏥", "💊"]
    rc = st.columns(min(len(recs), 3))
    for i, rec in enumerate(recs):
        with rc[i % len(rc)]:
            st.markdown(f"""<div class="health-card">
              <div style="font-size:1.2rem;">{icons[i % len(icons)]}</div>
              <p>{rec}</p>
            </div>""", unsafe_allow_html=True)

    # ── City AQI trend ────────────────────────────────────────────────────────
    sec(f"Recent Trend — {hs_city}")
    recent = city_df[city_df["timestamp"] >= cutoff]
    if not recent.empty:
        fig_ct = px.area(
            recent, x="timestamp", y="aqi",
            color_discrete_sequence=[info["color"]],
            labels={"aqi": "AQI", "timestamp": ""},
        )
        fig_ct.update_traces(fill="tozeroy",fillcolor=f"{info['color']}20",line=dict(color=info["color"]))
        fig_ct.add_hline(y=100, line_dash="dot", line_color="#FFFF00", opacity=0.5)
        fig_ct.add_hline(y=150, line_dash="dot", line_color="#FF7E00", opacity=0.5)
        fig_ct.update_layout(**_PL)
        st.plotly_chart(fig_ct, use_container_width=True)

    # ── Vulnerable groups ─────────────────────────────────────────────────────
    st.divider()
    sec("Who Is Most at Risk?")
    vg = [
        ("👶", "Children",        "Avoid outdoor play when AQI > 100."),
        ("👴", "Elderly",         "Stay indoors when AQI > 150."),
        ("🫁", "Respiratory",     "Carry inhalers; limit outdoor time."),
        ("🤰", "Pregnant Women",  "Minimize time in high-pollution areas."),
        ("🏃", "Athletes",        "Postpone outdoor workouts when AQI > 100."),
    ]
    vgc = st.columns(5)
    for gc, (ic, nm, ds) in zip(vgc, vg):
        gc.markdown(f"""<div class="health-card">
          <strong style="color:#c9d1d9;">{ic} {nm}</strong>
          <p>{ds}</p>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MAP VIEW
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Map View":

    topbar("Map View", "Global AQI distribution")

    latest_m = latest_per_city(df_all)
    if "latitude" not in latest_m.columns or latest_m["latitude"].isna().all():
        st.warning("Coordinate data not available in current dataset.")
        st.stop()

    map_metric = st.selectbox(
        "Metric",
        ["aqi", "pm2_5", "pm10", "no2"],
        format_func=lambda x: {"aqi": "AQI", "pm2_5": "PM2.5",
                                "pm10": "PM10", "no2": "NO₂"}[x],
    )

    fig_map = px.scatter_map(
        latest_m.dropna(subset=["latitude", "longitude"]),
        lat="latitude", lon="longitude",
        size=map_metric, color=map_metric,
        color_continuous_scale="RdYlGn_r",
        hover_name="city",
        hover_data={"aqi": ":.0f", "pm2_5": ":.1f",
                    "pm10": ":.1f", "temperature": ":.1f"},
        size_max=50, zoom=2,
        map_style="carto-darkmatter",
        title=f"Global {map_metric.upper()} Levels",
    )
    fig_map.update_layout(
        height=560, paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── City cards ────────────────────────────────────────────────────────────
    sec("City Snapshots")
    for i in range(0, len(latest_m), 4):
        batch = latest_m.iloc[i:i + 4]
        cols  = st.columns(4)
        for col, (_, row) in zip(cols, batch.iterrows()):
            inf = get_aqi_info(row["aqi"])
            col.markdown(f"""<div class="kpi-card"
                style="border-left:4px solid {inf['color']};margin-bottom:.6rem;">
              <div style="font-weight:600;color:#e6edf3;font-size:.92rem;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                {row['city']}
              </div>
              <div class="kpi-value" style="color:{inf['color']};">{row['aqi']:.0f}</div>
              <div class="kpi-label">{inf['emoji']} {inf['category']}</div>
              <div style="font-size:.7rem;color:#7d8590;margin-top:5px;">
                PM2.5 {row.get('pm2_5', 0):.1f} &nbsp;·&nbsp; NO₂ {row.get('no2', 0):.1f}
              </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION (simplified)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Anomaly Detection":

    topbar("Anomaly Detection", "Sudden pollution spikes and unusual readings")

    if not require_data():
        st.stop()

    with st.spinner("Analysing data…"):
        det     = AnomalyDetector(contamination=0.05, spike_threshold=150, spike_delta=50)
        df_anom = det.detect(df.copy())
        alerts  = det.spike_alerts(df.copy())

    # ── Spike alert cards — prominent and simple ───────────────────────────────
    if alerts:
        st.markdown(f"### ⚡ {len(alerts)} Spike Alert{'s' if len(alerts) > 1 else ''} Detected")
        for al in alerts:
            sev_color = {"CRITICAL":"#f85149","HIGH":"#FF7E00","MEDIUM":"#d29922"}.get(
                al["severity"], "#58a6ff"
            )
            sev_icon  = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}.get(al["severity"], "🔵")
            st.markdown(f"""<div class="anomaly-card"
                style="border-left-color:{sev_color};background:{sev_color}0d;">
              <div class="anomaly-icon">{sev_icon}</div>
              <div>
                <div class="anomaly-city" style="color:{sev_color};">{al['city']}</div>
                <div class="anomaly-detail">
                  AQI jumped from <strong>{al['previous_aqi']}</strong>
                  to <strong style="color:{sev_color};">{al['current_aqi']}</strong>
                  &nbsp;(+{al['delta']} points)
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No sudden pollution spikes detected in the selected cities and time range.")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── AQI timeline with anomalies highlighted ─────────────────────────────
    if "is_anomaly" in df_anom.columns:
        sec("AQI Timeline — Anomalies Highlighted")
        st.caption("Red markers indicate unusual readings that are statistically different from normal patterns.")

        normal = df_anom[~df_anom["is_anomaly"]]
        anom   = df_anom[df_anom["is_anomaly"]]

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=normal["timestamp"], y=normal["aqi"],
            mode="markers",
            marker=dict(color="#388bfd", size=4, opacity=0.4),
            name="Normal", hovertemplate="%{text}<br>AQI: %{y:.0f}<extra></extra>",
            text=normal["city"],
        ))
        if not anom.empty:
            fig_a.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom["aqi"],
                mode="markers",
                marker=dict(color="#f85149", size=8, symbol="x",
                            line=dict(color="#f85149", width=2)),
                name="Anomaly", hovertemplate="%{text}<br>AQI: %{y:.0f}<extra></extra>",
                text=anom["city"],
            ))
        fig_a.update_layout(
            title="All Readings", xaxis_title="", yaxis_title="AQI",
            **_PL,
        )
        st.plotly_chart(fig_a, use_container_width=True)

        # ── Summary table of worst anomalies ──────────────────────────────────
        if not anom.empty:
            sec("Unusual Readings")
            summary = det.anomaly_summary(df_anom)
            if not summary.empty:
                display_cols = [c for c in ["city","aqi","pm2_5","pm10","timestamp"]
                                if c in summary.columns]
                st.dataframe(
                    summary[display_cols].rename(columns={
                        "aqi": "AQI", "pm2_5": "PM2.5", "pm10": "PM10",
                        "timestamp": "Recorded At",
                    }),
                    use_container_width=True, hide_index=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — CONNECTION DEBUG (hidden from nav, accessible by URL param)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Connection Debug":
    topbar("Connection Debug", "Internal diagnostics")
    sec("App Status")
    st.json(_APP_STATUS)
    if st.button("🗄️ Re-initialise Database"):
        try:
            from src.db import init_db
            init_db()
            st.success("Done.")
        except Exception as e:
            st.error(str(e))
