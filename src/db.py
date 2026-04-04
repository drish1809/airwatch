"""
src/db.py — Database Abstraction Layer
========================================
Automatically uses:
  • PostgreSQL / Supabase  when DATABASE_URL is in Streamlit secrets or env var
  • SQLite                 when running locally without a cloud DB

Key fixes:
  • SSL enabled for Supabase (required by default)
  • Detailed error messages instead of silent failures
  • init_db() called on startup so the table always exists
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
# SQL schemas
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
# URL resolution
# ─────────────────────────────────────────────────────────────────────────────

def _get_pg_url() -> Optional[str]:
    """
    Returns the PostgreSQL connection URL from (in priority order):
      1. Streamlit secrets  →  st.secrets["database"]["url"]
      2. Environment var    →  DATABASE_URL
      3. None               →  use SQLite locally
    """
    # Streamlit secrets (works on Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "database" in st.secrets:
            url = st.secrets["database"].get("url", "")
            if url and url.startswith("postgres"):
                return url
    except Exception:
        pass

    # Environment variable (Render, Railway, Docker)
    env = os.environ.get("DATABASE_URL", "")
    if env and env.startswith("postgres"):
        return env

    return None


def is_postgres() -> bool:
    """True when a PostgreSQL URL is available."""
    return _get_pg_url() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Connection factory
# ─────────────────────────────────────────────────────────────────────────────

def _pg_connect():
    """
    Open a psycopg2 connection to Supabase.
    Supabase requires SSL — we enforce it here.
    """
    import psycopg2
    url = _get_pg_url()

    # Add sslmode=require if not already present (Supabase needs it)
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    try:
        return psycopg2.connect(url)
    except psycopg2.OperationalError as e:
        raise ConnectionError(
            f"Could not connect to Supabase.\n"
            f"Check your DATABASE_URL in Streamlit secrets.\n"
            f"Original error: {e}"
        ) from e


def _sqlite_path() -> Path:
    p = Path("data/air_quality.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the air_quality table if it does not already exist.
    Called once on app startup via _init_app().
    """
    if is_postgres():
        conn = _pg_connect()
        try:
            cur = conn.cursor()
            cur.execute(_CREATE_PG)
            conn.commit()
            logger.info("✅ Supabase table ready.")
        finally:
            conn.close()
    else:
        with sqlite3.connect(str(_sqlite_path())) as conn:
            conn.execute(_CREATE_SQLITE)
        logger.info("✅ SQLite table ready: %s", _sqlite_path())


def save_record(rec: dict) -> None:
    """
    Insert one air-quality reading. Silently skips duplicates.
    """
    vals = (
        str(rec["timestamp"]),
        rec["city"],    rec["country"],
        rec["latitude"], rec["longitude"],
        rec["aqi"],  rec["co"],   rec["no"],
        rec["no2"],  rec["o3"],   rec["so2"],
        rec["pm2_5"],rec["pm10"], rec["nh3"],
        rec["temperature"], rec["humidity"], rec["wind_speed"],
    )

    if is_postgres():
        conn = _pg_connect()
        try:
            cur = conn.cursor()
            cur.execute(_INSERT_PG, vals)
            conn.commit()
        finally:
            conn.close()
    else:
        with sqlite3.connect(str(_sqlite_path())) as conn:
            conn.execute(_INSERT_SQLITE, vals)


def load_dataframe() -> pd.DataFrame:
    """
    Return the full air_quality table as a DataFrame.
    Returns an empty DataFrame on any error (caller handles fallback).
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

        logger.info("Loaded %d records from %s.", len(df),
                    "Supabase" if is_postgres() else "SQLite")
        return df

    except Exception as exc:
        logger.error("load_dataframe failed: %s", exc)
        return pd.DataFrame()


def record_count() -> int:
    """Return total number of rows in the table."""
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
                return conn.execute(
                    "SELECT COUNT(*) FROM air_quality"
                ).fetchone()[0]
    except Exception:
        return 0


def test_connection() -> tuple[bool, str]:
    """
    Test the database connection and return (success, message).
    Used by the diagnostics panel in the sidebar.
    """
    try:
        if is_postgres():
            conn = _pg_connect()
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM air_quality")
            n = cur.fetchone()[0]
            conn.close()
            return True, f"Supabase connected — {n:,} records"
        else:
            n = record_count()
            return True, f"SQLite connected — {n:,} records"
    except Exception as exc:
        return False, str(exc)
