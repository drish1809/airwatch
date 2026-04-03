"""
scheduler.py — Background Data Collection for Streamlit Cloud
==============================================================
Uses st.cache_resource so the scheduler starts ONCE per app deployment
and continues running across all user sessions — even on Streamlit Cloud.

Usage (in app/main.py):
    from src.scheduler import start_background_scheduler
    start_background_scheduler(config)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global flag to prevent duplicate scheduler starts
_scheduler_thread: Optional[threading.Thread] = None
_scheduler_lock = threading.Lock()


def _collection_loop(config: dict, interval_minutes: int) -> None:
    """
    Infinite loop that calls the collector every `interval_minutes` minutes.
    Designed to run in a daemon thread so it dies with the process.
    """
    # Lazy import inside thread to avoid circular deps
    from src.collector import AirQualityCollector

    api_key = config["api"].get("openweather_api_key", "")
    if not api_key or api_key == "YOUR_OPENWEATHER_API_KEY":
        logger.warning("No valid API key set — background collector is paused.")
        return

    collector = AirQualityCollector(config)
    logger.info("🔄 Background collector started (interval=%d min).", interval_minutes)

    while True:
        try:
            logger.info("Background collection triggered…")
            collector.collect_all_cities()
        except Exception as exc:
            logger.error("Background collection error: %s", exc)

        time.sleep(interval_minutes * 60)


def start_background_scheduler(config: dict) -> None:
    """
    Start the background collection thread if it isn't already running.
    Safe to call multiple times (idempotent).

    Works on:
      - Local machines
      - Render / Fly.io
      - Streamlit Cloud (via st.cache_resource wrapper in app/main.py)
    """
    global _scheduler_thread

    with _scheduler_lock:
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            logger.debug("Background scheduler already running — skipping.")
            return

        interval = config.get("scheduler", {}).get("interval_minutes", 30)

        t = threading.Thread(
            target=_collection_loop,
            args=(config, interval),
            daemon=True,
            name="AQCollector",
        )
        t.start()
        _scheduler_thread = t
        logger.info("Background scheduler thread started (daemon=True).")
