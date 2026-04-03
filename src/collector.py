"""
collector.py — Automated Air Quality Data Collection
=====================================================
Fetches air quality + weather data from the OpenWeatherMap API for every
configured city, stores results in SQLite, and optionally exports to CSV.

A background APScheduler job fires this routine at a configurable interval
so the dataset grows continuously without manual intervention.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler

from src.db import init_db, save_record, load_dataframe

logger = logging.getLogger(__name__)

# Schema handled by src/db.py


# ─────────────────────────────────────────────────────────────────────────────
# Collector class
# ─────────────────────────────────────────────────────────────────────────────

class AirQualityCollector:
    """
    Fetches air quality + weather data from OpenWeatherMap and persists it.

    Usage
    -----
    collector = AirQualityCollector(config)
    collector.collect_all_cities()          # one-shot
    scheduler = collector.start_scheduler() # background loop
    """

    def __init__(self, config: dict) -> None:
        self.config       = config
        self.api_key      = config["api"]["openweather_api_key"]
        self.base_url     = config["api"]["base_url"]
        self.geo_url      = config["api"]["geocoding_url"]
        self.timeout      = config["api"].get("timeout_seconds", 10)
        self.db_path      = config["data"]["db_path"]
        self.raw_path     = config["data"]["raw_path"]
        self.cities       = config["cities"]
        self.max_retries  = config["scheduler"]["max_retries"]
        self.retry_delay  = config["scheduler"]["retry_delay_seconds"]
        self.rate_delay   = config["scheduler"].get("rate_limit_delay_seconds", 1.2)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.raw_path).mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Database ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        init_db()   # delegates to src/db.py — works for both SQLite and Postgres
        logger.info("Database initialised.")

    def _save_record(self, rec: dict) -> None:
        save_record(rec)   # delegates to src/db.py

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict) -> Optional[dict]:
        """GET with retry logic. Returns parsed JSON or None on failure."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Attempt %d/%d failed (%s): %s",
                               attempt, self.max_retries, url, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        logger.error("All %d attempts failed for %s", self.max_retries, url)
        return None

    # ── API calls ─────────────────────────────────────────────────────────────

    def get_coordinates(self, city: str, country: str) -> Optional[dict]:
        data = self._get(self.geo_url, {
            "q": f"{city},{country}", "limit": 1, "appid": self.api_key
        })
        if data:
            return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
        return None

    def fetch_air_quality(self, lat: float, lon: float) -> Optional[dict]:
        return self._get(f"{self.base_url}/air_pollution", {
            "lat": lat, "lon": lon, "appid": self.api_key
        })

    def fetch_weather(self, lat: float, lon: float) -> Optional[dict]:
        return self._get(f"{self.base_url}/weather", {
            "lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"
        })

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _owm_aqi_to_us(owm_aqi: int) -> float:
        """
        Map OpenWeatherMap AQI (1-5 scale) to a US-AQI-like value (0-500).
        Midpoints: 1→25, 2→75, 3→125, 4→175, 5→300
        """
        mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
        return float(mapping.get(owm_aqi, owm_aqi * 50))

    def _parse_record(self, aq_data: dict, weather_data: Optional[dict],
                      city: str, country: str,
                      lat: float, lon: float) -> Optional[dict]:
        try:
            entry      = aq_data["list"][0]
            components = entry["components"]
            owm_aqi    = entry["main"]["aqi"]
            ts         = datetime.fromtimestamp(entry["dt"])

            rec: dict = {
                "timestamp": ts,
                "city":      city,
                "country":   country,
                "latitude":  lat,
                "longitude": lon,
                "aqi":       self._owm_aqi_to_us(owm_aqi),
                "co":        components.get("co"),
                "no":        components.get("no"),
                "no2":       components.get("no2"),
                "o3":        components.get("o3"),
                "so2":       components.get("so2"),
                "pm2_5":     components.get("pm2_5"),
                "pm10":      components.get("pm10"),
                "nh3":       components.get("nh3"),
                "temperature": None,
                "humidity":    None,
                "wind_speed":  None,
            }

            if weather_data:
                rec["temperature"] = weather_data["main"].get("temp")
                rec["humidity"]    = weather_data["main"].get("humidity")
                rec["wind_speed"]  = weather_data["wind"].get("speed")

            return rec
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Parse error for %s: %s", city, exc)
            return None

    # ── Collection orchestration ──────────────────────────────────────────────

    def collect_all_cities(self) -> int:
        """
        Collect data for every configured city.
        Returns the count of successfully saved records.
        """
        logger.info("─── Starting collection run (%d cities) ───", len(self.cities))
        success = 0

        for city_cfg in self.cities:
            city    = city_cfg["name"]
            country = city_cfg["country"]

            coords = self.get_coordinates(city, country)
            if not coords:
                logger.error("Skipping %s — coordinates not found.", city)
                continue

            aq_data      = self.fetch_air_quality(coords["lat"], coords["lon"])
            weather_data = self.fetch_weather(coords["lat"], coords["lon"])

            if aq_data:
                rec = self._parse_record(aq_data, weather_data, city, country,
                                         coords["lat"], coords["lon"])
                if rec:
                    self._save_record(rec)
                    success += 1
                    logger.info("✓ %-14s | AQI=%5.1f | PM2.5=%5.2f | Temp=%s°C",
                                city, rec["aqi"], rec.get("pm2_5") or 0,
                                rec.get("temperature") or "N/A")

            time.sleep(self.rate_delay)   # be polite to the API

        logger.info("─── Collection done: %d/%d cities ───", success, len(self.cities))

        if self.config["data"].get("export_csv", True):
            self.export_to_csv()

        return success

    # ── CSV export ────────────────────────────────────────────────────────────

    def export_to_csv(self) -> str:
        """Export full database table to CSV and return the file path."""
        try:
            df = load_dataframe()
            csv_path = os.path.join(self.raw_path, "air_quality_data.csv")
            df.to_csv(csv_path, index=False)
            logger.info("Exported %d records → %s", len(df), csv_path)
            return csv_path
        except Exception as exc:
            logger.error("CSV export failed: %s", exc)
            return ""

    # ── Load helpers ──────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Return the full air_quality table as a DataFrame."""
        return load_dataframe()

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def start_scheduler(self, interval_minutes: Optional[int] = None) -> BackgroundScheduler:
        """
        Start an APScheduler background job that calls collect_all_cities()
        every `interval_minutes` minutes.

        The first run fires immediately (before the scheduler starts).
        """
        if interval_minutes is None:
            interval_minutes = self.config["scheduler"]["interval_minutes"]

        # Immediate first run
        logger.info("Running initial collection before scheduler starts…")
        self.collect_all_cities()

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            func=self.collect_all_cities,
            trigger="interval",
            minutes=interval_minutes,
            id="aq_collection",
            name="Air Quality Collection",
            misfire_grace_time=120,
        )
        scheduler.start()
        logger.info("Scheduler started — collecting every %d min.", interval_minutes)
        return scheduler
