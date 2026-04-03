"""
utils.py — Shared helper utilities
Covers: config loading, logging setup, directory management, AQI helpers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load YAML config, then override with Streamlit secrets if on cloud.
    Priority: 1) Streamlit secrets  2) Env var OWM_API_KEY  3) config.yaml
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r") as fh:
        config = yaml.safe_load(fh)

    # Override with Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "api" in st.secrets:
                key = st.secrets["api"].get("openweather_api_key", "")
                if key:
                    config["api"]["openweather_api_key"] = key
    except Exception:
        pass

    # Override with environment variable (Render / Docker)
    env_key = os.environ.get("OWM_API_KEY", "")
    if env_key:
        config["api"]["openweather_api_key"] = env_key

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO",
                  log_file: str = "logs/app.log",
                  max_bytes: int = 10_485_760,
                  backup_count: int = 5) -> logging.Logger:
    """
    Configure root logger with rotating file + console handlers.
    Returns the root logger.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not root.handlers:          # avoid duplicate handlers on reload
        root.addHandler(fh)
        root.addHandler(ch)

    return root


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(paths: list[str]) -> None:
    """Create directories (and parents) if they do not already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# AQI helpers
# ─────────────────────────────────────────────────────────────────────────────

# Lookup table: (upper_bound, category, hex_color, emoji)
_AQI_LEVELS: list[tuple[int, str, str, str]] = [
    (50,  "Good",                            "#00E400", "✅"),
    (100, "Moderate",                        "#FFFF00", "⚠️"),
    (150, "Unhealthy for Sensitive Groups",  "#FF7E00", "🟠"),
    (200, "Unhealthy",                       "#FF0000", "🔴"),
    (300, "Very Unhealthy",                  "#8F3F97", "🟣"),
    (500, "Hazardous",                       "#7E0023", "☠️"),
]

_HEALTH_ADVICE: dict[str, dict[str, Any]] = {
    "Good": {
        "health_msg": "Air quality is satisfactory — enjoy outdoor activities!",
        "recommendations": [
            "Perfect for outdoor exercise and activities.",
            "No special precautions needed.",
            "Enjoy fresh air — open your windows!",
        ],
    },
    "Moderate": {
        "health_msg": "Acceptable air quality. Sensitive individuals should be cautious.",
        "recommendations": [
            "Sensitive groups (asthma, elderly) reduce prolonged outdoor exertion.",
            "Consider a basic mask if you are sensitive.",
            "Keep windows partially open.",
        ],
    },
    "Unhealthy for Sensitive Groups": {
        "health_msg": "Sensitive groups may experience health effects. Wear a mask.",
        "recommendations": [
            "Wear an N95 mask outdoors.",
            "Children, elderly and those with respiratory conditions should stay indoors.",
            "Keep windows and doors closed.",
            "Use an air purifier indoors.",
        ],
    },
    "Unhealthy": {
        "health_msg": "Everyone may experience health effects. Limit outdoor exposure.",
        "recommendations": [
            "Wear an N95/N99 mask outdoors.",
            "Limit outdoor activities to short durations.",
            "Keep windows and doors closed.",
            "Run a HEPA air purifier continuously.",
            "Avoid strenuous outdoor exercise.",
        ],
    },
    "Very Unhealthy": {
        "health_msg": "Health alert! Avoid all outdoor activities.",
        "recommendations": [
            "Stay indoors with windows sealed.",
            "Run air purifier continuously.",
            "Wear an N99 mask even in transit.",
            "Children and elderly must NOT go outside.",
            "Consider a medical checkup if symptomatic.",
        ],
    },
    "Hazardous": {
        "health_msg": "EMERGENCY: Extreme pollution — stay indoors and seek help if unwell.",
        "recommendations": [
            "STAY INDOORS — Emergency air quality.",
            "Seal doors, windows and any gaps.",
            "Use multiple air purifiers simultaneously.",
            "Wear N99 mask even indoors if possible.",
            "Seek medical attention for any respiratory symptoms immediately.",
        ],
    },
}


def get_aqi_info(aqi: float) -> dict[str, Any]:
    """
    Return a dict with category, color, emoji, health_msg, and recommendations
    for the given AQI value.
    """
    for upper, category, color, emoji in _AQI_LEVELS:
        if aqi <= upper:
            advice = _HEALTH_ADVICE.get(category, {})
            return {
                "category":       category,
                "color":          color,
                "emoji":          emoji,
                "health_msg":     advice.get("health_msg", ""),
                "recommendations": advice.get("recommendations", []),
            }
    # Fallback for AQI > 500
    advice = _HEALTH_ADVICE["Hazardous"]
    return {
        "category":       "Hazardous",
        "color":          "#7E0023",
        "emoji":          "☠️",
        "health_msg":     advice["health_msg"],
        "recommendations": advice["recommendations"],
    }


def aqi_category(aqi: float) -> str:
    """Return only the AQI category string."""
    return get_aqi_info(aqi)["category"]
