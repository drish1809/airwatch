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
    "Delhi":       {"base": 185, "var": 60,  "lat": 28.66,  "lon":  77.21,  "country": "IN"},
    "Beijing":     {"base": 150, "var": 70,  "lat": 39.91,  "lon": 116.39,  "country": "CN"},
    "Mumbai":      {"base": 110, "var": 40,  "lat": 19.08,  "lon":  72.88,  "country": "IN"},
    "Kolkata":     {"base": 130, "var": 50,  "lat": 22.57,  "lon":  88.36,  "country": "IN"},
    "Chennai":     {"base": 90,  "var": 35,  "lat": 13.08,  "lon":  80.27,  "country": "IN"},
    "Bangalore":   {"base": 80,  "var": 30,  "lat": 12.97,  "lon":  77.59,  "country": "IN"},
    "Los Angeles": {"base": 100, "var": 45,  "lat": 34.05,  "lon":-118.24,  "country": "US"},
    "New York":    {"base": 65,  "var": 30,  "lat": 40.71,  "lon": -74.01,  "country": "US"},
    "London":      {"base": 55,  "var": 25,  "lat": 51.51,  "lon":  -0.13,  "country": "GB"},
    "Paris":       {"base": 60,  "var": 28,  "lat": 48.86,  "lon":   2.35,  "country": "FR"},
    "Tokyo":       {"base": 50,  "var": 20,  "lat": 35.69,  "lon": 139.69,  "country": "JP"},
    "Sydney":      {"base": 40,  "var": 18,  "lat":-33.87,  "lon": 151.21,  "country": "AU"},
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
