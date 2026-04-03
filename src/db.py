"""
src/db.py — Database Abstraction Layer
========================================
Automatically uses:
  - PostgreSQL (Supabase) when DATABASE_URL is in Streamlit secrets or env
  - SQLite                when running locally without a cloud DB

This means the same code works locally AND on Streamlit Cloud
with zero changes between environments.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS air_quality (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL,
    city        TEXT     NOT NULL,
    country     TEXT,
    latitude    REAL, longitude  REAL,
    aqi         REAL, co        REAL, no   REAL,
    no2         REAL, o3        REAL, so2  REAL,
    pm2_5       REAL, pm10      REAL, nh3  REAL,
    temperature REAL, humidity  REAL, wind_speed REAL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timestamp, city)
)
"""

_CREATE_PG = """
CREATE TABLE IF NOT EXISTS air_quality (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    city        TEXT        NOT NULL,
    country     TEXT,
    latitude    REAL, longitude  REAL,
    aqi         REAL, co        REAL, no   REAL,
    no2         REAL, o3        REAL, so2  REAL,
    pm2_5       REAL, pm10      REAL, nh3  REAL,
    temperature REAL, humidity  REAL, wind_speed REAL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, city)
)
"""

_INSERT_SQLITE = """
INSERT OR IGNORE INTO air_quality
  (timestamp,city,country,latitude,longitude,
   aqi,co,no,no2,o3,so2,pm2_5,pm10,nh3,
   temperature,humidity,wind_speed)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_INSERT_PG = """
INSERT INTO air_quality
  (timestamp,city,country,latitude,longitude,
   aqi,co,no,no2,o3,so2,pm2_5,pm10,nh3,
   temperature,humidity,wind_speed)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (timestamp,city) DO NOTHING
"""


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pg_url() -> Optional[str]:
    """
    Return Postgres connection URL from:
    1. Streamlit secrets  →  [database] url
    2. Environment var    →  DATABASE_URL
    3. None               →  fall back to SQLite
    """
    # Try Streamlit secrets (works on Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "database" in st.secrets:
            url = st.secrets["database"].get("url", "")
            if url:
                return url
    except Exception:
        pass

    # Try environment variable (works on Render / Railway / etc.)
    return os.environ.get("DATABASE_URL") or None


def is_postgres() -> bool:
    return _get_pg_url() is not None


def _pg_connect():
    """Open a psycopg2 connection to Supabase / Postgres."""
    import psycopg2
    return psycopg2.connect(_get_pg_url())


def _sqlite_path() -> Path:
    p = Path("data/air_quality.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the air_quality table if it does not already exist."""
    if is_postgres():
        conn = _pg_connect()
        try:
            with conn:
                conn.cursor().execute(_CREATE_PG)
            logger.info("PostgreSQL table ready.")
        finally:
            conn.close()
    else:
        with sqlite3.connect(str(_sqlite_path())) as conn:
            conn.execute(_CREATE_SQLITE)
        logger.info("SQLite table ready: %s", _sqlite_path())


def save_record(rec: dict) -> None:
    """
    Insert one air-quality record. Silently skips duplicates.
    rec keys: timestamp, city, country, latitude, longitude,
              aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3,
              temperature, humidity, wind_speed
    """
    values = (
        str(rec["timestamp"]),
        rec["city"], rec["country"],
        rec["latitude"], rec["longitude"],
        rec["aqi"],   rec["co"],   rec["no"],
        rec["no2"],   rec["o3"],   rec["so2"],
        rec["pm2_5"], rec["pm10"], rec["nh3"],
        rec["temperature"], rec["humidity"], rec["wind_speed"],
    )

    if is_postgres():
        conn = _pg_connect()
        try:
            with conn:
                conn.cursor().execute(_INSERT_PG, values)
        finally:
            conn.close()
    else:
        with sqlite3.connect(str(_sqlite_path())) as conn:
            conn.execute(_INSERT_SQLITE, values)


def load_dataframe() -> pd.DataFrame:
    """
    Return the full air_quality table as a pandas DataFrame.
    Returns an empty DataFrame if the table is empty or doesn't exist yet.
    """
    try:
        if is_postgres():
            conn = _pg_connect()
            try:
                df = pd.read_sql(
                    "SELECT * FROM air_quality ORDER BY city, timestamp",
                    conn,
                )
            finally:
                conn.close()
        else:
            db = _sqlite_path()
            if not db.exists():
                return pd.DataFrame()
            with sqlite3.connect(str(db)) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM air_quality ORDER BY city, timestamp",
                    conn,
                )

        logger.debug("Loaded %d records from DB.", len(df))
        return df

    except Exception as exc:
        logger.error("load_dataframe failed: %s", exc)
        return pd.DataFrame()


def record_count() -> int:
    """Return total number of records in the table."""
    try:
        if is_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM air_quality")
                return cur.fetchone()[0]
            finally:
                conn.close()
        else:
            db = _sqlite_path()
            if not db.exists():
                return 0
            with sqlite3.connect(str(db)) as conn:
                return conn.execute("SELECT COUNT(*) FROM air_quality").fetchone()[0]
    except Exception:
        return 0
