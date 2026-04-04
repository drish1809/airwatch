"""
demo_data.py — Synthetic Data Generator
========================================
Creates a realistic 30-day air quality dataset so the full Streamlit app
can be demonstrated without an OpenWeatherMap API key.

Run:  python -m src.demo_data
"""

from __future__ import annotations

import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── City profiles (base AQI, variability, lat, lon) ────────────────────────
CITIES = {
    # India
    "Delhi":           {"base":185,"var":65,"lat": 28.66,"lon":  77.21,"country":"IN"},
    "Mumbai":          {"base":115,"var":40,"lat": 19.08,"lon":  72.88,"country":"IN"},
    "Bangalore":       {"base": 82,"var":30,"lat": 12.97,"lon":  77.59,"country":"IN"},
    "Chennai":         {"base": 92,"var":35,"lat": 13.08,"lon":  80.27,"country":"IN"},
    "Kolkata":         {"base":132,"var":50,"lat": 22.57,"lon":  88.36,"country":"IN"},
    "Hyderabad":       {"base": 95,"var":35,"lat": 17.38,"lon":  78.49,"country":"IN"},
    "Pune":            {"base": 88,"var":30,"lat": 18.52,"lon":  73.86,"country":"IN"},
    "Ahmedabad":       {"base":118,"var":42,"lat": 23.02,"lon":  72.57,"country":"IN"},
    "Jaipur":          {"base":138,"var":48,"lat": 26.91,"lon":  75.79,"country":"IN"},
    "Surat":           {"base":102,"var":38,"lat": 21.17,"lon":  72.83,"country":"IN"},
    # China
    "Beijing":         {"base":155,"var":70,"lat": 39.91,"lon": 116.39,"country":"CN"},
    "Shanghai":        {"base":105,"var":45,"lat": 31.23,"lon": 121.47,"country":"CN"},
    "Guangzhou":       {"base": 95,"var":40,"lat": 23.13,"lon": 113.26,"country":"CN"},
    "Shenzhen":        {"base": 88,"var":35,"lat": 22.55,"lon": 114.07,"country":"CN"},
    "Chengdu":         {"base":120,"var":50,"lat": 30.57,"lon": 104.07,"country":"CN"},
    # Asia
    "Tokyo":           {"base": 52,"var":20,"lat": 35.69,"lon": 139.69,"country":"JP"},
    "Seoul":           {"base": 85,"var":35,"lat": 37.57,"lon": 126.98,"country":"KR"},
    "Bangkok":         {"base":112,"var":42,"lat": 13.75,"lon": 100.52,"country":"TH"},
    "Singapore":       {"base": 48,"var":18,"lat":  1.35,"lon": 103.82,"country":"SG"},
    "Kuala Lumpur":    {"base": 72,"var":28,"lat":  3.14,"lon": 101.69,"country":"MY"},
    "Jakarta":         {"base":138,"var":50,"lat": -6.21,"lon": 106.85,"country":"ID"},
    "Manila":          {"base":125,"var":45,"lat": 14.60,"lon": 120.98,"country":"PH"},
    "Karachi":         {"base":175,"var":60,"lat": 24.86,"lon":  67.01,"country":"PK"},
    "Lahore":          {"base":188,"var":65,"lat": 31.55,"lon":  74.35,"country":"PK"},
    "Dhaka":           {"base":165,"var":58,"lat": 23.72,"lon":  90.41,"country":"BD"},
    "Kathmandu":       {"base":145,"var":52,"lat": 27.71,"lon":  85.31,"country":"NP"},
    "Colombo":         {"base": 68,"var":25,"lat":  6.93,"lon":  79.85,"country":"LK"},
    "Hanoi":           {"base":118,"var":42,"lat": 21.03,"lon": 105.85,"country":"VN"},
    "Ho Chi Minh City":{"base":108,"var":40,"lat": 10.82,"lon": 106.63,"country":"VN"},
    # Middle East
    "Dubai":           {"base": 95,"var":38,"lat": 25.20,"lon":  55.27,"country":"AE"},
    "Riyadh":          {"base":108,"var":42,"lat": 24.69,"lon":  46.72,"country":"SA"},
    "Tehran":          {"base":142,"var":52,"lat": 35.69,"lon":  51.39,"country":"IR"},
    "Istanbul":        {"base": 85,"var":32,"lat": 41.01,"lon":  28.97,"country":"TR"},
    "Baghdad":         {"base":148,"var":55,"lat": 33.34,"lon":  44.40,"country":"IQ"},
    # Europe
    "London":          {"base": 55,"var":22,"lat": 51.51,"lon":  -0.13,"country":"GB"},
    "Paris":           {"base": 62,"var":25,"lat": 48.86,"lon":   2.35,"country":"FR"},
    "Berlin":          {"base": 52,"var":20,"lat": 52.52,"lon":  13.40,"country":"DE"},
    "Madrid":          {"base": 65,"var":25,"lat": 40.42,"lon":  -3.70,"country":"ES"},
    "Rome":            {"base": 70,"var":28,"lat": 41.90,"lon":  12.48,"country":"IT"},
    "Amsterdam":       {"base": 50,"var":20,"lat": 52.37,"lon":   4.90,"country":"NL"},
    "Brussels":        {"base": 58,"var":22,"lat": 50.85,"lon":   4.35,"country":"BE"},
    "Vienna":          {"base": 48,"var":18,"lat": 48.21,"lon":  16.37,"country":"AT"},
    "Warsaw":          {"base": 78,"var":30,"lat": 52.23,"lon":  21.01,"country":"PL"},
    "Stockholm":       {"base": 38,"var":15,"lat": 59.33,"lon":  18.07,"country":"SE"},
    "Moscow":          {"base": 88,"var":35,"lat": 55.75,"lon":  37.62,"country":"RU"},
    "Athens":          {"base": 72,"var":28,"lat": 37.98,"lon":  23.73,"country":"GR"},
    "Lisbon":          {"base": 55,"var":22,"lat": 38.72,"lon":  -9.14,"country":"PT"},
    "Zurich":          {"base": 38,"var":14,"lat": 47.38,"lon":   8.54,"country":"CH"},
    # Africa
    "Cairo":           {"base":172,"var":62,"lat": 30.06,"lon":  31.25,"country":"EG"},
    "Lagos":           {"base":158,"var":58,"lat":  6.52,"lon":   3.38,"country":"NG"},
    "Nairobi":         {"base": 72,"var":28,"lat": -1.29,"lon":  36.82,"country":"KE"},
    "Casablanca":      {"base": 82,"var":30,"lat": 33.59,"lon":  -7.62,"country":"MA"},
    "Johannesburg":    {"base": 75,"var":28,"lat":-26.20,"lon":  28.04,"country":"ZA"},
    # Americas North
    "New York":        {"base": 65,"var":28,"lat": 40.71,"lon": -74.01,"country":"US"},
    "Los Angeles":     {"base":102,"var":42,"lat": 34.05,"lon":-118.24,"country":"US"},
    "Chicago":         {"base": 72,"var":30,"lat": 41.88,"lon": -87.63,"country":"US"},
    "Toronto":         {"base": 55,"var":22,"lat": 43.65,"lon": -79.38,"country":"CA"},
    "Mexico City":     {"base":148,"var":52,"lat": 19.43,"lon": -99.13,"country":"MX"},
    "Houston":         {"base": 85,"var":35,"lat": 29.76,"lon": -95.37,"country":"US"},
    "Phoenix":         {"base": 92,"var":38,"lat": 33.45,"lon":-112.07,"country":"US"},
    "Miami":           {"base": 62,"var":25,"lat": 25.77,"lon": -80.19,"country":"US"},
    "Vancouver":       {"base": 42,"var":17,"lat": 49.25,"lon":-123.12,"country":"CA"},
    "Montreal":        {"base": 52,"var":20,"lat": 45.51,"lon": -73.55,"country":"CA"},
    # Americas South
    "Sao Paulo":       {"base":118,"var":42,"lat":-23.55,"lon": -46.63,"country":"BR"},
    "Buenos Aires":    {"base": 82,"var":30,"lat":-34.61,"lon": -58.38,"country":"AR"},
    "Bogota":          {"base": 95,"var":38,"lat":  4.71,"lon": -74.07,"country":"CO"},
    "Lima":            {"base": 88,"var":35,"lat":-12.05,"lon": -77.04,"country":"PE"},
    "Santiago":        {"base": 98,"var":38,"lat":-33.46,"lon": -70.65,"country":"CL"},
    "Rio de Janeiro":  {"base": 88,"var":35,"lat":-22.91,"lon": -43.17,"country":"BR"},
    # Oceania
    "Sydney":          {"base": 42,"var":18,"lat":-33.87,"lon": 151.21,"country":"AU"},
    "Melbourne":       {"base": 38,"var":16,"lat":-37.81,"lon": 144.96,"country":"AU"},
    "Auckland":        {"base": 35,"var":14,"lat":-36.87,"lon": 174.77,"country":"NZ"},
}

HOURS_PER_DAY   = 24
INTERVAL_HOURS  = 0.5   # record every 30 min
DAYS_BACK       = 30
rng             = np.random.default_rng(seed=42)


def _rush_hour_multiplier(hour: int) -> float:
    """Traffic peaks at 8 AM and 6 PM."""
    if 7 <= hour <= 9:
        return 1.25
    if 17 <= hour <= 19:
        return 1.18
    if 0 <= hour <= 5:
        return 0.75
    return 1.0


def _generate_city_series(city: str, profile: dict,
                           timestamps: list[datetime]) -> pd.DataFrame:
    n   = len(timestamps)
    base = profile["base"]
    var  = profile["var"]

    # Correlated AQI with AR(1) process
    eps   = rng.normal(0, var * 0.3, n)
    trend = np.zeros(n)
    trend[0] = base + eps[0]
    for i in range(1, n):
        trend[i] = 0.85 * trend[i - 1] + 0.15 * base + eps[i]

    # Diurnal pattern
    hours = np.array([ts.hour for ts in timestamps])
    rush  = np.vectorize(_rush_hour_multiplier)(hours)
    aqi   = np.clip(trend * rush, 5, 500)

    pm2_5  = aqi * rng.uniform(0.35, 0.45, n)
    pm10   = pm2_5 * rng.uniform(1.5, 2.0, n)
    no2    = pm2_5 * rng.uniform(0.3, 0.6, n)
    co     = pm2_5 * rng.uniform(2.0, 5.0, n)
    o3     = rng.uniform(30, 120, n)
    so2    = pm2_5 * rng.uniform(0.05, 0.15, n)
    no_col = no2   * rng.uniform(0.1, 0.3, n)
    nh3    = pm2_5 * rng.uniform(0.02, 0.08, n)

    temp    = rng.uniform(18, 36, n)
    humidity= rng.uniform(30, 90, n)
    wind    = rng.uniform(0, 15, n)

    return pd.DataFrame({
        "timestamp":   timestamps,
        "city":        city,
        "country":     profile["country"],
        "latitude":    profile["lat"],
        "longitude":   profile["lon"],
        "aqi":         np.round(aqi,   1),
        "co":          np.round(co,    3),
        "no":          np.round(no_col,3),
        "no2":         np.round(no2,   3),
        "o3":          np.round(o3,    3),
        "so2":         np.round(so2,   3),
        "pm2_5":       np.round(pm2_5, 2),
        "pm10":        np.round(pm10,  2),
        "nh3":         np.round(nh3,   3),
        "temperature": np.round(temp,  1),
        "humidity":    np.round(humidity, 1),
        "wind_speed":  np.round(wind,  2),
    })


def generate(db_path: str = "data/air_quality.db",
             csv_path: str = "data/raw/air_quality_data.csv") -> pd.DataFrame:
    """Generate synthetic data and store in SQLite + CSV."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    end_ts   = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_ts = end_ts - timedelta(days=DAYS_BACK)
    steps    = int(DAYS_BACK * 24 / INTERVAL_HOURS)
    timestamps = [start_ts + timedelta(hours=i * INTERVAL_HOURS) for i in range(steps + 1)]

    frames = [
        _generate_city_series(city, profile, timestamps)
        for city, profile in CITIES.items()
    ]
    df = pd.concat(frames, ignore_index=True)

    # Write SQLite
    _CREATE = """
    CREATE TABLE IF NOT EXISTS air_quality (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME, city TEXT, country TEXT,
        latitude REAL, longitude REAL,
        aqi REAL, co REAL, no REAL, no2 REAL, o3 REAL,
        so2 REAL, pm2_5 REAL, pm10 REAL, nh3 REAL,
        temperature REAL, humidity REAL, wind_speed REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(timestamp, city)
    )"""
    _INS = """
    INSERT OR IGNORE INTO air_quality
      (timestamp,city,country,latitude,longitude,
       aqi,co,no,no2,o3,so2,pm2_5,pm10,nh3,
       temperature,humidity,wind_speed)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    # SQLite needs plain strings, not pandas Timestamps
    df_ins = df.copy()
    df_ins["timestamp"] = df_ins["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE)
        conn.executemany(_INS, df_ins[[
            "timestamp","city","country","latitude","longitude",
            "aqi","co","no","no2","o3","so2","pm2_5","pm10","nh3",
            "temperature","humidity","wind_speed"
        ]].values.tolist())

    df.to_csv(csv_path, index=False)
    print(f"✓ Generated {len(df):,} synthetic records for {len(CITIES)} cities.")
    print(f"  SQLite → {db_path}")
    print(f"  CSV    → {csv_path}")
    return df


if __name__ == "__main__":
    generate()
