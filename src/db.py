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
# Connection parameter resolution
# Supports BOTH individual params AND a URL — individual params are preferred
# because they have zero URL-parsing issues.
# ─────────────────────────────────────────────────────────────────────────────

def _get_pg_params() -> Optional[dict]:
    """
    Returns psycopg2 connection keyword arguments, or None for SQLite.

    Checks (in order):
      1. Streamlit secrets — individual fields [database] host/password/etc.
      2. Streamlit secrets — single URL        [database] url
      3. Environment var  — DATABASE_URL
      4. None             — fall back to SQLite
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "database" in st.secrets:
            sec = st.secrets["database"]

            # ── Option A: individual fields (recommended, zero parsing issues) ──
            host = sec.get("host", "")
            if host and "supabase" in host:
                return {
                    "host":     host,
                    "port":     int(sec.get("port", 5432)),
                    "dbname":   sec.get("dbname", "postgres"),
                    "user":     sec.get("user", "postgres"),
                    "password": str(sec.get("password", "")),
                    "sslmode":  "require",
                }

            # ── Option B: single URL string ────────────────────────────────────
            url = sec.get("url", "")
            if url and url.startswith("postgres"):
                return {"dsn": url + ("&sslmode=require" if "?" in url
                                      else "?sslmode=require")}
    except Exception:
        pass

    # Environment variable fallback
    env = os.environ.get("DATABASE_URL", "")
    if env and env.startswith("postgres"):
        return {"dsn": env + ("&sslmode=require" if "?" in env else "?sslmode=require")}

    return None   # → use SQLite


def is_postgres() -> bool:
    """True when Supabase / Postgres credentials are available."""
    return _get_pg_params() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Connection factory
# ─────────────────────────────────────────────────────────────────────────────

def _pg_connect():
    """
    Open a psycopg2 connection using individual keyword arguments.
    No URL parsing — completely avoids URL format issues.
    """
    import psycopg2
    params = _get_pg_params()

    if params is None:
        raise ConnectionError(
            "No Postgres credentials found.\n"
            "Add [database] section to Streamlit secrets."
        )

    try:
        return psycopg2.connect(**params)
    except psycopg2.OperationalError as e:
        raise ConnectionError(
            f"Supabase connection failed: {e}\n"
            "Check the 🔧 Connection Debug page for help."
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
