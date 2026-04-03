"""
app/main.py — AirWatch Dashboard v3
=====================================
Targeted fixes over v2:
  ✅ [1]  Persistent AirWatch brand header on every page
  ✅ [2]  Correct region filtering: World=all, Country=strict, India=IN only
  ✅ [3]  No "showing all cities" fallback — empty region shows clear message & stops
  ✅ [4]  City Comparison uses full dataset, not sidebar subset
  ✅ [5]  Prediction: auto-prefills live data + "Enter manually" checkbox
  ✅ [6]  Health & Safety: same cascading filter, strict data intersection
  ✅ [7]  Clean spacing, consistent typography, minimal clutter
  ✅ [8]  All pages use same cached data source; loading spinners added
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
from src.cities import (
    get_all_countries, get_cities_for_country,
    get_india_states, get_india_cities_for_state,
    get_all_india_cities,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWatch — Real-Time Air Quality",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Background scheduler — starts once per server process, survives all sessions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_scheduler():
    try:
        cfg = load_config(str(ROOT / "config" / "config.yaml"))
        from src.scheduler import start_background_scheduler
        start_background_scheduler(cfg)
    except Exception:
        pass
    return True

_init_scheduler()

# ─────────────────────────────────────────────────────────────────────────────
# Global styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e14 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 6px 0; }

/* ── Brand header ── */
.airwatch-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.1rem 1.6rem;
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.airwatch-logo { font-size: 2rem; line-height: 1; }
.airwatch-title { font-size: 1.75rem; font-weight: 700; color: #e6edf3; letter-spacing: -.02em; }
.airwatch-title span { color: #58a6ff; }
.airwatch-subtitle { font-size: .8rem; color: #7d8590; margin-top: 2px; }
.airwatch-badge {
    margin-left: auto;
    font-size: .7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #3fb950;
    border: 1px solid rgba(63,185,80,.35);
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
}
.airwatch-badge.demo { color: #d29922; border-color: rgba(210,153,34,.35); }

/* ── Page sub-header ── */
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 .25rem;
    letter-spacing: -.01em;
}
.page-subtitle { font-size: .85rem; color: #7d8590; margin-bottom: 1.5rem; }

/* ── Section label ── */
.section-label {
    font-size: 1rem;
    font-weight: 600;
    color: #c9d1d9;
    border-left: 3px solid #58a6ff;
    padding-left: .65rem;
    margin: 1.75rem 0 .9rem;
}

/* ── KPI card ── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    transition: border-color .18s, transform .18s;
}
.kpi-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
.kpi-value {
    font-size: 1.75rem; font-weight: 700;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}
.kpi-label {
    font-size: .68rem; color: #7d8590;
    text-transform: uppercase; letter-spacing: .06em; margin-top: 5px;
}

/* ── AQI badge ── */
.aqi-badge {
    display: inline-block;
    padding: 5px 18px; border-radius: 20px;
    font-weight: 600; font-size: .88rem; letter-spacing: .03em;
}

/* ── Health card ── */
.health-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: .9rem 1.1rem;
    height: 100%;
}
.health-card p { color: #8b949e; font-size: .84rem; margin: .4rem 0 0; }

/* ── Prediction placeholder ── */
.predict-placeholder {
    background: #161b22; border: 2px dashed #21262d;
    border-radius: 14px; padding: 4rem 2rem;
    text-align: center; color: #7d8590;
}
.predict-placeholder .icon { font-size: 2.5rem; margin-bottom: .75rem; }
.predict-placeholder p { font-size: 1rem; margin: 0; }

/* ── Alert box ── */
.spike-alert {
    border-radius: 10px; padding: .9rem 1.2rem; margin: .4rem 0;
    border-left-width: 4px; border-left-style: solid;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: rgba(88,166,255,.1);
    border: 1px solid rgba(88,166,255,.25);
    color: #58a6ff;
    font-size: .75rem;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: .5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Misc ── */
.stMetric [data-testid="metric-container"] {
    background: #161b22; border-radius: 10px; padding: 12px; border: 1px solid #21262d;
}
div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
hr { border-color: #21262d !important; margin: 1.5rem 0; }
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
# Data layer — single cached source used by every page
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load data from:
      1. Supabase / PostgreSQL  (when DATABASE_URL secret is set — Streamlit Cloud)
      2. SQLite                 (when running locally)
      3. Demo CSV               (fallback so app is never blank)
    Adds all derived columns used across pages.
    """
    from src.db import load_dataframe
    df = load_dataframe()

    if df.empty:
        # Final fallback: demo CSV bundled in the repo
        csv = ROOT / "data" / "raw" / "air_quality_data.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            st.caption("ℹ️ Showing demo data. Connect a database for live data.")
        else:
            st.error("⛔ No data found. Run `python run.py --demo` to generate demo data.")
            st.stop()

    df["timestamp"]    = pd.to_datetime(df["timestamp"])
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
    """Return one (latest) row per city."""
    return df.sort_values("timestamp").groupby("city").last().reset_index()


def get_city_latest(df_all: pd.DataFrame, city: str) -> dict:
    """Return the most recent row for a city as a plain dict (safe defaults)."""
    rows = df_all[df_all["city"] == city].sort_values("timestamp")
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {col: (float(row[col]) if pd.notna(row[col]) else 0.0)
            for col in ["pm2_5", "pm10", "no2", "o3", "so2", "co",
                        "temperature", "humidity", "wind_speed", "aqi"]}


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def brand_header(page_name: str, subtitle: str, live: bool) -> None:
    """Render the persistent AirWatch brand header at the top of every page."""
    badge_cls  = "airwatch-badge" if live else "airwatch-badge demo"
    badge_text = "● LIVE" if live else "● DEMO"
    st.markdown(f"""
    <div class="airwatch-header">
      <div class="airwatch-logo">🌬️</div>
      <div>
        <div class="airwatch-title">Air<span>Watch</span></div>
        <div class="airwatch-subtitle">Real-Time Air Quality Intelligence</div>
      </div>
      <div style="margin-left:1.5rem; border-left:1px solid #21262d; padding-left:1.5rem;">
        <div style="font-size:.7rem;color:#7d8590;text-transform:uppercase;letter-spacing:.07em;">Current View</div>
        <div style="font-size:.95rem;font-weight:600;color:#c9d1d9;margin-top:2px;">{page_name}</div>
        <div style="font-size:.75rem;color:#7d8590;margin-top:1px;">{subtitle}</div>
      </div>
      <div class="{badge_cls}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


def kpi(col, value: str, label: str, color: str = "#58a6ff") -> None:
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Region → cities resolution  (FIX #2 & #3)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_cities(region_mode: str, country_sel: str, state_sel: str,
                   all_cities_in_data: list[str]) -> list[str]:
    """
    Strict resolution — never falls back to showing all cities.
    Returns only cities that exist in the dataset for the chosen region.
    Returns an empty list if no match, letting the caller decide.
    """
    if region_mode == "🌍 All Cities":
        return all_cities_in_data                          # every city in DB

    if region_mode == "🇮🇳 India":
        india_pool = (get_all_india_cities() if state_sel == "All States"
                      else get_india_cities_for_state(state_sel))
        return [c for c in india_pool if c in all_cities_in_data]

    # Specific country
    country_pool = get_cities_for_country(country_sel)
    return [c for c in country_pool if c in all_cities_in_data]


# ─────────────────────────────────────────────────────────────────────────────
# Load data & predictor (top-level — used everywhere)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_all    = load_data()
    predictor = load_predictor()

all_cities_in_data = sorted(df_all["city"].unique().tolist())

# API key status (used in header badge)
try:
    _cfg    = load_config(str(ROOT / "config" / "config.yaml"))
    _apikey = _cfg["api"].get("openweather_api_key", "")
    IS_LIVE = bool(_apikey and _apikey != "YOUR_OPENWEATHER_API_KEY")
except Exception:
    IS_LIVE = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand in sidebar
    st.markdown("""
    <div style="padding:.5rem 0 1rem;">
      <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;letter-spacing:-.01em;">
        🌬️ Air<span style="color:#58a6ff;">Watch</span>
      </div>
      <div style="font-size:.72rem;color:#7d8590;margin-top:2px;">Real-Time Air Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Navigation
    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🌍 City Comparison",
        "🔮 AQI Prediction",
        "⚠️ Health & Safety",
        "🗺️ Map View",
        "🔬 Anomaly Detection",
    ], label_visibility="collapsed")

    st.divider()

    # ── Region selector ───────────────────────────────────────────────────────
    st.markdown("**🌐 Region**")
    region_mode = st.radio("region_mode", [
        "🌍 All Cities",
        "🇮🇳 India",
        "🌏 Country",
    ], label_visibility="collapsed")

    country_sel = "India"
    state_sel   = "All States"

    if region_mode == "🇮🇳 India":
        state_sel = st.selectbox("State / UT", ["All States"] + get_india_states(),
                                 key="sb_state")
    elif region_mode == "🌏 Country":
        country_sel = st.selectbox("Country", get_all_countries(),
                                   index=get_all_countries().index("India"),
                                   key="sb_country")

    # Resolve which cities apply to this region
    region_cities = resolve_cities(region_mode, country_sel, state_sel, all_cities_in_data)

    # ── City multiselect (only cities that exist in data) ────────────────────
    st.markdown("**🏙️ Cities**")
    if not region_cities:
        st.warning("No data collected for this region yet.\nTry **All Cities** or a different country.")
        selected_cities = []
    else:
        selected_cities = st.multiselect(
            "cities", region_cities,
            default=region_cities[:min(6, len(region_cities))],
            label_visibility="collapsed",
            placeholder="Choose cities…",
        )
        if not selected_cities:
            selected_cities = region_cities   # default to all in region

    # ── Time range ────────────────────────────────────────────────────────────
    st.markdown("**📅 Time Range**")
    days_back = st.slider("days", 1, 30, 7, label_visibility="collapsed")
    cutoff    = datetime.now() - timedelta(days=days_back)

    # Filtered dataframe used by Dashboard, Anomaly, Map
    df = (df_all[
        (df_all["city"].isin(selected_cities)) &
        (df_all["timestamp"] >= cutoff)
    ].copy() if selected_cities else pd.DataFrame())

    st.divider()

    # ── Status & controls ─────────────────────────────────────────────────────
    if IS_LIVE:
        st.success("🔄 Live collection is **ON**", icon=None)
    else:
        st.caption("⚠️ Demo mode — add your OpenWeather API key in `config/config.yaml` to enable live data collection.")

    st.caption(f"Data: {len(df_all):,} records · {len(all_cities_in_data)} cities")
    st.caption(f"Refreshed: {datetime.now():%H:%M:%S}")

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st_autorefresh(interval=600_000, key="autorefresh")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: stop if no cities selected (applies to Dashboard / Anomaly)
# ─────────────────────────────────────────────────────────────────────────────
def require_cities() -> bool:
    if not selected_cities:
        brand_header(page, "No region data", IS_LIVE)
        st.warning("No data found for the selected region. "
                   "Please choose **All Cities** or switch to a region that has been collected.")
        return False
    if df.empty:
        brand_header(page, "No data in range", IS_LIVE)
        st.warning("No records for this date range. Try extending the time window in the sidebar.")
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    brand_header(
        "Dashboard",
        f"{len(selected_cities)} cities · last {days_back} {'day' if days_back == 1 else 'days'}",
        IS_LIVE,
    )

    if not require_cities():
        st.stop()

    latest = latest_per_city(df)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    worst_city = latest.loc[latest["aqi"].idxmax(), "city"]
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{latest['aqi'].mean():.0f}",
         f"{latest['pm2_5'].mean():.1f}",
         f"{latest['pm10'].mean():.1f}",
         f"{latest['no2'].mean():.1f}",
         worst_city],
        ["Avg AQI", "Avg PM2.5 μg/m³", "Avg PM10 μg/m³", "Avg NO₂ μg/m³", "Most Polluted City"],
    ):
        kpi(col, val, lbl)

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Bar + Donut ───────────────────────────────────────────────────────────
    col_bar, col_pie = st.columns([3, 2])

    with col_bar:
        section("Pollution Ranking")
        top = latest.sort_values("aqi", ascending=False)
        fig_bar = go.Figure(go.Bar(
            x=top["city"], y=top["aqi"],
            marker_color=[AQI_CM.get(c, "#58a6ff") for c in top["aqi_category"]],
            text=top["aqi"].round(0).astype(int),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>",
        ))
        fig_bar.update_layout(showlegend=False, **_PL)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        section("AQI Category Mix")
        cats = latest["aqi_category"].value_counts().reset_index()
        cats.columns = ["category", "count"]
        fig_pie = px.pie(cats, values="count", names="category",
                         color="category", color_discrete_map=AQI_CM, hole=0.5)
        fig_pie.update_traces(textinfo="label+percent",
                              textfont_size=11, pull=[0.04] * len(cats))
        fig_pie.update_layout(**_PL)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── AQI Trend ─────────────────────────────────────────────────────────────
    section("AQI Trend Over Time")
    trend_cities = selected_cities[:10]   # cap for legibility
    fig_trend = px.line(
        df[df["city"].isin(trend_cities)],
        x="timestamp", y="aqi", color="city",
        labels={"aqi": "AQI", "timestamp": ""},
    )
    fig_trend.add_hline(y=100, line_dash="dash", line_color="#FFFF00",
                        opacity=0.6, annotation_text="Moderate")
    fig_trend.add_hline(y=200, line_dash="dash", line_color="#FF0000",
                        opacity=0.6, annotation_text="Unhealthy")
    fig_trend.update_layout(**_PL)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Hourly Heatmap ────────────────────────────────────────────────────────
    section("Hourly PM2.5 Heatmap")
    pivot = (df.groupby(["city", "hour"])["pm2_5"].mean()
               .reset_index()
               .pivot(index="city", columns="hour", values="pm2_5"))
    if not pivot.empty:
        fig_h = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                          labels={"x": "Hour of Day", "y": "City", "color": "PM2.5"})
        fig_h.update_layout(**_PL)
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 Latest readings per city"):
        show_cols = [c for c in ["city","aqi","aqi_category","pm2_5","pm10","no2",
                                  "o3","temperature","humidity","timestamp"]
                     if c in latest.columns]
        st.dataframe(
            latest[show_cols].sort_values("aqi", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CITY COMPARISON  (FIX #4: uses all_cities_in_data, not sidebar subset)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌍 City Comparison":

    brand_header("City Comparison", "Compare pollution profiles side by side", IS_LIVE)

    # Always compare across the full dataset — independent of sidebar region
    compare = st.multiselect(
        "Select 2 to 8 cities to compare",
        all_cities_in_data,
        default=all_cities_in_data[:min(4, len(all_cities_in_data))],
        placeholder="Search for a city…",
    )
    if len(compare) < 2:
        st.info("Please select at least 2 cities to compare.")
        st.stop()

    df_cmp  = df_all[df_all["city"].isin(compare) & (df_all["timestamp"] >= cutoff)]
    lat_cmp = latest_per_city(df_cmp)

    if df_cmp.empty:
        st.warning("No data for the selected cities in this date range. Try extending the time window.")
        st.stop()

    # ── Metric picker ─────────────────────────────────────────────────────────
    met = st.selectbox(
        "Primary metric",
        ["aqi", "pm2_5", "pm10", "no2", "temperature", "humidity"],
        format_func=lambda x: {
            "aqi": "AQI", "pm2_5": "PM2.5 (μg/m³)", "pm10": "PM10 (μg/m³)",
            "no2": "NO₂ (μg/m³)", "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
        }[x],
    )

    # ── Radar chart ───────────────────────────────────────────────────────────
    section("Pollutant Radar — Normalised")
    mr = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
    lr = ["PM2.5", "PM10", "NO₂", "O₃", "SO₂", "CO"]
    fig_r = go.Figure()
    for _, row in lat_cmp.iterrows():
        vals = [float(row.get(m) or 0) for m in mr]
        maxv = [max(float(lat_cmp[m].max()), 1.0) for m in mr]
        norm = [v / mx * 100 for v, mx in zip(vals, maxv)]
        fig_r.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=lr + [lr[0]],
            fill="toself", name=row["city"], opacity=0.72,
        ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1c2128"),
            angularaxis=dict(gridcolor="#1c2128"),
        ), **_PL,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Side-by-side trend ────────────────────────────────────────────────────
    section(f"{met.upper()} Trend — Side by Side")
    fig_facet = px.line(
        df_cmp, x="timestamp", y=met, color="city",
        facet_col="city", facet_col_wrap=min(4, len(compare)),
        labels={met: met.upper(), "timestamp": ""},
    )
    fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_facet.update_layout(**_PL)
    st.plotly_chart(fig_facet, use_container_width=True)

    # ── Statistical summary ───────────────────────────────────────────────────
    section("Statistical Summary")
    sum_cols = [c for c in ["aqi", "pm2_5", "pm10", "no2", "temperature"]
                if c in df_cmp.columns]
    summary = df_cmp.groupby("city")[sum_cols].agg(["mean", "max", "min"]).round(1)
    summary.columns = [f"{c[0].replace('_',' ').upper()} ({c[1]})" for c in summary.columns]
    st.dataframe(summary, use_container_width=True)

    # ── Box plot ──────────────────────────────────────────────────────────────
    section("AQI Distribution")
    fig_box = px.box(df_cmp, x="city", y="aqi", color="city",
                     labels={"aqi": "AQI"}, points="outliers")
    fig_box.update_layout(showlegend=False, **_PL)
    st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AQI PREDICTION  (FIX #5: auto-prefill + manual override)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 AQI Prediction":

    brand_header("AQI Prediction", "ML-powered forecast for any city & hour", IS_LIVE)

    if not predictor.is_ready:
        st.warning("⚠️ No trained model found. Run `python run.py --train` first. "
                   "Using a simple heuristic estimate in the meantime.")

    # ── Step 1: City selection ────────────────────────────────────────────────
    st.markdown('<div class="info-pill">STEP 1 — Select Location</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        pred_country = st.selectbox("Country", get_all_countries(),
                                    index=get_all_countries().index("India"),
                                    key="pred_country")
    with pc2:
        if pred_country == "India":
            pred_state = st.selectbox("State / UT",
                                      ["Any State"] + get_india_states(),
                                      key="pred_state")
        else:
            pred_state = "Any State"
            st.selectbox("State / UT", ["—"], disabled=True, key="pred_state_dis")
    with pc3:
        if pred_country == "India" and pred_state != "Any State":
            city_choices = get_india_cities_for_state(pred_state)
        else:
            city_choices = get_cities_for_country(pred_country)
        pred_city = st.selectbox("City", city_choices or ["—"], key="pred_city_sel")

    # ── Step 2: Date & hour ───────────────────────────────────────────────────
    st.markdown('<div class="info-pill">STEP 2 — Select Date & Hour</div>', unsafe_allow_html=True)
    pt1, pt2 = st.columns(2)
    with pt1:
        pred_hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    with pt2:
        pred_date = st.date_input("Date", datetime.today())

    # ── Step 3: Pollutant inputs ──────────────────────────────────────────────
    # Try to fetch latest data for the chosen city
    city_live = get_city_latest(df_all, pred_city)
    has_live  = bool(city_live)

    st.markdown('<div class="info-pill">STEP 3 — Pollutant Inputs</div>', unsafe_allow_html=True)

    if has_live:
        st.caption(
            f"✅ Live data available for **{pred_city}** — values pre-filled from latest reading. "
            "Toggle below to enter custom values."
        )
    else:
        st.caption(f"ℹ️ No collected data for **{pred_city}**. Enter values manually.")

    manual_override = st.checkbox(
        "Enter values manually",
        value=not has_live,
        help="Uncheck to use the latest collected data for this city as input.",
    )

    # Determine default values
    if has_live and not manual_override:
        v_pm25  = city_live.get("pm2_5",  35.0)
        v_pm10  = city_live.get("pm10",   65.0)
        v_no2   = city_live.get("no2",    28.0)
        v_temp  = city_live.get("temperature", 28.0)
        v_humid = city_live.get("humidity",    65.0)
    else:
        v_pm25, v_pm10, v_no2, v_temp, v_humid = 35.0, 65.0, 28.0, 28.0, 65.0

    pp1, pp2, pp3, pp4, pp5 = st.columns(5)
    disabled_inputs = (has_live and not manual_override)

    with pp1: pm25  = st.number_input("PM2.5 (μg/m³)",  0.0, 500.0, float(round(v_pm25, 1)),  step=1.0, disabled=disabled_inputs)
    with pp2: pm10  = st.number_input("PM10 (μg/m³)",   0.0, 600.0, float(round(v_pm10, 1)),  step=1.0, disabled=disabled_inputs)
    with pp3: no2   = st.number_input("NO₂ (μg/m³)",    0.0, 300.0, float(round(v_no2,  1)),  step=1.0, disabled=disabled_inputs)
    with pp4: temp  = st.number_input("Temperature (°C)", -20.0, 55.0, float(round(v_temp, 1)),  step=0.5, disabled=disabled_inputs)
    with pp5: humid = st.number_input("Humidity (%)",    0.0, 100.0, float(round(v_humid, 1)), step=1.0, disabled=disabled_inputs)

    if disabled_inputs:
        st.caption(f"Using values from latest {pred_city} reading. Enable manual entry above to override.")

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("")
    clicked = st.button("🔮 Predict AQI Now", type="primary", use_container_width=True)

    if not clicked:
        st.markdown("""<div class="predict-placeholder">
          <div class="icon">🔮</div>
          <p>Configure the location and inputs above,<br>then click <b>Predict AQI Now</b>.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Compute prediction ────────────────────────────────────────────────────
    pred_day   = pd.Timestamp(pred_date).dayofweek
    is_weekend = int(pred_day >= 5)
    is_rush    = int(pred_hour in [7, 8, 9, 17, 18, 19])
    city_enc   = all_cities_in_data.index(pred_city) if pred_city in all_cities_in_data else 0

    features = {
        "hour": pred_hour, "day_of_week": pred_day,
        "month": pd.Timestamp(pred_date).month,
        "is_weekend": is_weekend, "is_rush_hour": is_rush,
        "pm2_5": pm25, "pm10": pm10, "no2": no2,
        "temperature": temp, "humidity": humid,
        "pm_ratio":      pm25 / (pm10 + 1e-6),
        "no_no2_ratio":  no2  * 0.2 / (no2 + 1e-6),
        "pm2_5_lag_1": pm25 * 0.95, "pm2_5_lag_2": pm25 * 0.90, "pm2_5_lag_3": pm25 * 0.87,
        "aqi_lag_1": pm25 * 3.5,    "aqi_lag_2": pm25 * 3.3,    "aqi_lag_3": pm25 * 3.1,
        "pm2_5_rolling_mean_3": pm25, "pm2_5_rolling_std_3": pm25 * 0.1,
        "aqi_rolling_mean_3": pm25 * 3.4,
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

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    section(f"Prediction Result — {pred_city}  ·  {pred_date}  {pred_hour:02d}:00")

    rg, rr = st.columns([1, 1])
    with rg:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_aqi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted AQI", "font": {"size": 15}},
            number={"font": {"size": 48, "color": info["color"]}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": "#c9d1d9"},
                "bar": {"color": info["color"]},
                "steps": [
                    {"range": [0,   50],  "color": "#002200"},
                    {"range": [50,  100], "color": "#222200"},
                    {"range": [100, 150], "color": "#221500"},
                    {"range": [150, 200], "color": "#220000"},
                    {"range": [200, 300], "color": "#1a0022"},
                    {"range": [300, 500], "color": "#1a0008"},
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
          <p style="color:#8b949e;text-align:center;font-size:.88rem;margin:0 0 1rem;">
            {info['health_msg']}
          </p>
          <hr>
          <div style="font-weight:600;color:#c9d1d9;margin-bottom:.5rem;">Recommended Precautions</div>
          <ul style="padding-left:1.1rem;margin:0;">
            {"".join(f"<li style='color:#8b949e;font-size:.84rem;margin:5px 0;'>{r}</li>" for r in info['recommendations'])}
          </ul>
        </div>""", unsafe_allow_html=True)

        if predictor.is_ready:
            m = predictor.metrics
            st.caption(f"Model: **{predictor.model_name}** · RMSE {m.get('rmse','?')} · R² {m.get('r2','?')}")

    # ── 24-hour forecast ──────────────────────────────────────────────────────
    section("24-Hour Forecast")
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
    fig_fc.add_trace(go.Scatter(
        x=fc_df["hour"], y=fc_df["aqi"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2.5),
        marker=dict(size=7, color=[AQI_CM.get(c, "#58a6ff") for c in fc_df["cat"]],
                    line=dict(color="white", width=1)),
        fill="tozeroy", fillcolor="rgba(88,166,255,.06)",
        hovertemplate="%{x}:00 — AQI: %{y:.0f}<extra></extra>",
    ))
    for yv, col, lab in [(100, "#FFFF00", "Moderate"), (150, "#FF7E00", "Sensitive"), (200, "#FF0000", "Unhealthy")]:
        fig_fc.add_hline(y=yv, line_dash="dot", line_color=col, opacity=0.5, annotation_text=lab)
    fig_fc.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Predicted AQI",
        title=f"24-Hour AQI Forecast — {pred_city}", **_PL,
    )
    st.plotly_chart(fig_fc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HEALTH & SAFETY  (FIX #6: cascading + strict data intersection)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Health & Safety":

    brand_header("Health & Safety", "City-specific AQI health guidance", IS_LIVE)

    # ── Location selector (cascading, same pattern as prediction) ─────────────
    st.markdown('<div class="info-pill">SELECT LOCATION</div>', unsafe_allow_html=True)
    hs1, hs2, hs3 = st.columns(3)

    with hs1:
        hs_country = st.selectbox("Country", get_all_countries(),
                                  index=get_all_countries().index("India"),
                                  key="hs_country")
    with hs2:
        if hs_country == "India":
            hs_state = st.selectbox("State / UT",
                                    ["Any State"] + get_india_states(),
                                    key="hs_state")
        else:
            hs_state = "Any State"
            st.selectbox("State / UT", ["—"], disabled=True, key="hs_state_dis")

    with hs3:
        if hs_country == "India" and hs_state != "Any State":
            hs_pool = get_india_cities_for_state(hs_state)
        else:
            hs_pool = get_cities_for_country(hs_country)

        # Strict intersection — only cities that exist in the dataset
        hs_avail = [c for c in hs_pool if c in all_cities_in_data]

        if not hs_avail:
            st.selectbox("City", ["No data available"], disabled=True, key="hs_city_dis")
            st.warning(
                f"No collected data for cities in this region. "
                "Try a different country/state, or run `python run.py --demo` to seed data."
            )
            st.stop()

        hs_city = st.selectbox("City", hs_avail, key="hs_city_sel")

    st.markdown("---")

    # ── City data ─────────────────────────────────────────────────────────────
    city_df  = df_all[df_all["city"] == hs_city].sort_values("timestamp")
    last_row = city_df.iloc[-1]
    aqi_val  = float(last_row["aqi"])
    info     = get_aqi_info(aqi_val)

    # ── KPI row ───────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4 = st.columns(4)
    for col, val, lbl in zip(
        [kc1, kc2, kc3, kc4],
        [f"{aqi_val:.0f}",
         f"{float(last_row.get('pm2_5') or 0):.1f}",
         f"{float(last_row.get('pm10') or 0):.1f}",
         f"{float(last_row.get('no2') or 0):.1f}"],
        ["AQI", "PM2.5 μg/m³", "PM10 μg/m³", "NO₂ μg/m³"],
    ):
        kpi(col, val, lbl, color=info["color"])

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── AQI badge ─────────────────────────────────────────────────────────────
    st.markdown(f"""<div style="text-align:center;margin-bottom:1.25rem;">
      <span class="aqi-badge"
        style="background:{info['color']}1a;color:{info['color']};border:1px solid {info['color']}88;font-size:1rem;">
        {info['emoji']} &nbsp; {info['category']}
      </span>
      <p style="color:#7d8590;margin:.6rem 0 0;font-size:.88rem;">{info['health_msg']}</p>
    </div>""", unsafe_allow_html=True)

    # ── Precautions ───────────────────────────────────────────────────────────
    section("Recommended Precautions")
    recs  = info["recommendations"]
    icons = ["😷", "🏠", "💨", "🌿", "🏥", "💊"]
    rec_cols = st.columns(min(len(recs), 3))
    for i, rec in enumerate(recs):
        with rec_cols[i % len(rec_cols)]:
            st.markdown(f"""<div class="health-card">
              <div style="font-size:1.25rem;">{icons[i % len(icons)]}</div>
              <p>{rec}</p>
            </div>""", unsafe_allow_html=True)

    # ── City trend ────────────────────────────────────────────────────────────
    section(f"AQI Trend — {hs_city}")
    recent_df = city_df[city_df["timestamp"] >= cutoff]
    if not recent_df.empty:
        fig_trend = px.line(recent_df, x="timestamp", y="aqi",
                            color_discrete_sequence=[info["color"]],
                            labels={"aqi": "AQI", "timestamp": ""})
        fig_trend.add_hline(y=100, line_dash="dot", line_color="#FFFF00", opacity=0.5)
        fig_trend.add_hline(y=150, line_dash="dot", line_color="#FF7E00", opacity=0.5)
        fig_trend.update_layout(**_PL)
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Pollutant bars ────────────────────────────────────────────────────────
    section("Current Pollutant Levels")
    poll_keys = [p for p in ["pm2_5", "pm10", "no2", "o3", "so2", "co", "nh3"] if p in last_row]
    poll_vals = [float(last_row[p] or 0) for p in poll_keys]
    fig_poll  = go.Figure(go.Bar(
        x=poll_keys, y=poll_vals,
        marker_color=[info["color"]] * len(poll_keys),
        text=[f"{v:.2f}" for v in poll_vals], textposition="outside",
    ))
    fig_poll.update_layout(showlegend=False, **_PL)
    st.plotly_chart(fig_poll, use_container_width=True)

    # ── Vulnerable groups ─────────────────────────────────────────────────────
    st.markdown("---")
    section("Who Is Most at Risk?")
    vg = [
        ("👶", "Children",       "Developing lungs are highly susceptible. Avoid outdoor play when AQI > 100."),
        ("👴", "Elderly",        "Stay indoors and use air purifiers when AQI > 150."),
        ("🫁", "Respiratory",    "Asthma & COPD patients: carry inhalers; limit outdoor exposure."),
        ("🤰", "Pregnant Women", "Minimize time in high-pollution areas to protect fetal development."),
        ("🏃", "Athletes",       "Postpone outdoor workouts when AQI > 100; breathe more deeply during exercise."),
    ]
    vgcols = st.columns(5)
    for gc, (ic, nm, ds) in zip(vgcols, vg):
        gc.markdown(f"""<div class="health-card">
          <strong style="color:#c9d1d9;">{ic} {nm}</strong>
          <p>{ds}</p>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MAP VIEW
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Map View":

    brand_header("Map View", "Geographic AQI distribution", IS_LIVE)

    latest_m = latest_per_city(df_all)   # always all cities for map
    if "latitude" not in latest_m.columns or latest_m["latitude"].isna().all():
        st.warning("Coordinate data not available in current dataset.")
        st.stop()

    map_metric = st.selectbox(
        "Metric to visualise",
        ["aqi", "pm2_5", "pm10", "no2"],
        format_func=lambda x: {"aqi": "AQI Index", "pm2_5": "PM2.5",
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
        size_max=50, zoom=2, map_style="carto-darkmatter",
        title=f"Global {map_metric.upper()} Levels",
    )
    fig_map.update_layout(
        height=580, paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── City cards ────────────────────────────────────────────────────────────
    section("City Snapshots")
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
# PAGE 6 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Anomaly Detection":

    brand_header("Anomaly Detection", "Spike alerts & statistical outliers", IS_LIVE)

    if not require_cities():
        st.stop()

    st.caption(
        "Isolation Forest flags the most anomalous 5% of readings. "
        "Spike alerts fire when AQI jumps ≥ 50 units between two consecutive readings."
    )

    det = AnomalyDetector(contamination=0.05, spike_threshold=150, spike_delta=50)
    with st.spinner("Running Isolation Forest…"):
        df_anom = det.detect(df.copy())
        alerts  = det.spike_alerts(df.copy())

    # ── Spike alerts ──────────────────────────────────────────────────────────
    section("⚡ Spike Alerts")
    if alerts:
        for al in alerts:
            sev_color = {"CRITICAL": "#f85149", "HIGH": "#FF7E00", "MEDIUM": "#d29922"}.get(
                al["severity"], "#58a6ff"
            )
            st.markdown(f"""<div class="spike-alert"
                style="border-color:{sev_color};background:{sev_color}0f;">
              <strong style="color:{sev_color};">[{al['severity']}] {al['city']}</strong>
              &nbsp;—&nbsp; AQI moved from
              <code>{al['previous_aqi']}</code> to <code>{al['current_aqi']}</code>
              &nbsp;(<b style="color:{sev_color};">+{al['delta']}</b>)
              &nbsp;·&nbsp; <span style="color:#7d8590;font-size:.8rem;">{al['timestamp']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No sudden spikes detected in the selected cities and time range.")

    # ── Anomaly scatter ───────────────────────────────────────────────────────
    section("Anomalous Readings")
    if "is_anomaly" in df_anom.columns:
        fig_a = px.scatter(
            df_anom, x="timestamp", y="aqi",
            color="is_anomaly",
            color_discrete_map={True: "#f85149", False: "#388bfd"},
            symbol="is_anomaly",
            symbol_map={True: "x", False: "circle"},
            opacity=0.65,
            hover_data=["city", "pm2_5", "pm10"],
            labels={"is_anomaly": "Anomaly"},
            title="AQI Readings — Anomalies Highlighted in Red",
        )
        fig_a.update_layout(**_PL)
        st.plotly_chart(fig_a, use_container_width=True)

        summary = det.anomaly_summary(df_anom)
        if not summary.empty:
            section("Top Anomalous Records")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Score distribution ────────────────────────────────────────────────────
    if "anomaly_score" in df_anom.columns:
        section("Anomaly Score Distribution")
        fig_sc = px.histogram(
            df_anom, x="anomaly_score", color="is_anomaly",
            color_discrete_map={True: "#f85149", False: "#388bfd"},
            nbins=50, barmode="overlay",
            labels={"anomaly_score": "Anomaly Score (higher = more anomalous)"},
        )
        fig_sc.update_traces(opacity=0.7)
        fig_sc.update_layout(**_PL)
        st.plotly_chart(fig_sc, use_container_width=True)
