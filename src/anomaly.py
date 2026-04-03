"""
anomaly.py — Pollution Anomaly Detection
==========================================
Uses Isolation Forest to flag statistical outliers in the collected dataset
and a simple delta-based heuristic to generate real-time spike alerts.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detect sudden pollution spikes and statistical anomalies.

    Usage
    -----
    detector = AnomalyDetector(contamination=0.05)
    df       = detector.detect(df)          # adds is_anomaly & anomaly_score columns
    alerts   = detector.spike_alerts(df)    # list of city-level spike dicts
    """

    _DEFAULT_FEATURES = ["pm2_5", "pm10", "no2", "o3", "aqi"]

    def __init__(
        self,
        contamination: float = 0.05,
        spike_threshold: float = 150.0,
        spike_delta: float = 50.0,
        random_state: int = 42,
    ) -> None:
        self.contamination   = contamination
        self.spike_threshold = spike_threshold
        self.spike_delta     = spike_delta
        self._iforest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Isolation Forest
    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit Isolation Forest and add two columns to df:
        * ``is_anomaly``   (bool)
        * ``anomaly_score`` (float, higher = more anomalous)
        """
        df = df.copy()
        cols = [c for c in (features or self._DEFAULT_FEATURES) if c in df.columns]

        if not cols or len(df) < 20:
            logger.warning("Insufficient data for anomaly detection. Skipping.")
            df["is_anomaly"]    = False
            df["anomaly_score"] = 0.0
            return df

        sub = df[cols].dropna()
        if len(sub) < 10:
            df["is_anomaly"]    = False
            df["anomaly_score"] = 0.0
            return df

        preds  = self._iforest.fit_predict(sub)          # +1 normal, -1 anomaly
        scores = -self._iforest.score_samples(sub)        # higher → more anomalous

        df["is_anomaly"]    = False
        df["anomaly_score"] = 0.0
        df.loc[sub.index, "is_anomaly"]    = preds == -1
        df.loc[sub.index, "anomaly_score"] = scores

        n_anomalies = int(df["is_anomaly"].sum())
        logger.info(
            "Anomaly detection: %d/%d records flagged (%.1f%%)",
            n_anomalies, len(df), 100 * n_anomalies / len(df),
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Spike alerts
    # ─────────────────────────────────────────────────────────────────────────

    def spike_alerts(self, df: pd.DataFrame) -> list[dict]:
        """
        Compare latest vs previous AQI reading per city.
        Returns a list of alert dicts for cities that exceed the thresholds.

        Each alert dict has keys:
        city, current_aqi, previous_aqi, delta, severity, timestamp
        """
        alerts: list[dict] = []

        for city, grp in df.groupby("city"):
            grp = grp.sort_values("timestamp")
            if len(grp) < 2:
                continue

            latest   = grp.iloc[-1]
            previous = grp.iloc[-2]

            current_aqi  = float(latest["aqi"])
            previous_aqi = float(previous["aqi"])
            delta        = current_aqi - previous_aqi

            if current_aqi >= self.spike_threshold and delta >= self.spike_delta:
                severity = "CRITICAL" if current_aqi >= 300 else \
                           "HIGH"     if current_aqi >= 200 else "MEDIUM"
                alerts.append({
                    "city":         city,
                    "current_aqi":  round(current_aqi,  1),
                    "previous_aqi": round(previous_aqi, 1),
                    "delta":        round(delta,         1),
                    "severity":     severity,
                    "timestamp":    str(latest["timestamp"]),
                })
                logger.info(
                    "⚠ Spike alert [%s] %s: AQI %.0f → %.0f (+%.0f)",
                    severity, city, previous_aqi, current_aqi, delta,
                )

        return alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    def anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with the top anomalous records (for dashboard display).
        Expects 'is_anomaly' and 'anomaly_score' columns (call detect() first).
        """
        if "is_anomaly" not in df.columns:
            return pd.DataFrame()

        anom = df[df["is_anomaly"]].sort_values(
            "anomaly_score", ascending=False
        )
        display_cols = [
            c for c in
            ["timestamp", "city", "aqi", "pm2_5", "pm10", "anomaly_score"]
            if c in anom.columns
        ]
        return anom[display_cols].head(20).reset_index(drop=True)
