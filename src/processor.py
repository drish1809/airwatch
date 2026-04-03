"""
processor.py — Data Processing & Feature Engineering Pipeline
=============================================================
Cleans raw data collected from SQLite, engineers time and domain features,
computes rolling/lag statistics, and packages everything for ML training.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import aqi_category

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    End-to-end processing pipeline:
        load → clean → feature engineer → ML-ready export

    Attributes
    ----------
    label_encoder : fitted LabelEncoder for the 'city' column.
    feature_columns : list of feature names used for ML (set after prepare_ml_features).
    """

    def __init__(self) -> None:
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.feature_columns: Optional[list[str]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_from_db(self, db_path: str) -> pd.DataFrame:
        """Load raw records from SQLite into a DataFrame."""
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM air_quality ORDER BY city, timestamp", conn
            )
        logger.info("Loaded %d raw records from %s", len(df), db_path)
        return df

    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load data from a CSV file (e.g., demo/seed data)."""
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logger.info("Loaded %d records from %s", len(df), csv_path)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Cleaning
    # ─────────────────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Steps
        -----
        1. Parse timestamps
        2. Drop exact duplicates on (timestamp, city)
        3. Impute missing numerics with city-median → global-median fallback
        4. Drop rows where AQI is still missing after imputation
        5. Winsorise extreme outliers (±3σ) for key pollutants
        """
        logger.info("Cleaning data (input shape: %s)…", df.shape)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Deduplication
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp", "city"])
        logger.info("Removed %d duplicate rows.", before - len(df))

        # Impute numeric columns
        numeric_cols = [
            "co", "no", "no2", "o3", "so2",
            "pm2_5", "pm10", "nh3",
            "temperature", "humidity", "wind_speed",
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
            # City-level median
            df[col] = df.groupby("city")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Global fallback
            df[col] = df[col].fillna(df[col].median())

        # Must have AQI
        df = df.dropna(subset=["aqi"])

        # Winsorise ±3σ for important columns
        for col in ["pm2_5", "pm10", "co", "no2", "o3"]:
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                df[col] = df[col].clip(mu - 3 * sigma, mu + 3 * sigma)

        logger.info("Clean data shape: %s", df.shape)
        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds
        ----
        * Calendar features: hour, day_of_week, month, is_weekend, is_rush_hour
        * Categorical: time_of_day, aqi_category
        * Ratio features: pm_ratio, no_no2_ratio
        * Lag features: pm2_5 and aqi lags 1-3 (per city)
        * Rolling: 3-point rolling mean/std for pm2_5 and aqi (per city)
        """
        logger.info("Engineering features…")
        df = df.sort_values(["city", "timestamp"]).copy()

        # ── Calendar ─────────────────────────────────────────────────────────
        df["hour"]         = df["timestamp"].dt.hour
        df["day_of_week"]  = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"]        = df["timestamp"].dt.month
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True,
        ).astype(str)

        # ── AQI label ────────────────────────────────────────────────────────
        df["aqi_category"] = df["aqi"].apply(aqi_category)

        # ── Ratio features ───────────────────────────────────────────────────
        df["pm_ratio"]     = df["pm2_5"] / (df["pm10"] + 1e-6)
        df["no_no2_ratio"] = df["no"]    / (df["no2"]  + 1e-6)

        # ── Lag features (group by city) ──────────────────────────────────────
        for lag in [1, 2, 3]:
            df[f"pm2_5_lag_{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
            df[f"aqi_lag_{lag}"]   = df.groupby("city")["aqi"].shift(lag)

        # ── Rolling statistics ────────────────────────────────────────────────
        for col in ["pm2_5", "aqi"]:
            df[f"{col}_rolling_mean_3"] = (
                df.groupby("city")[col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean())
            )
            df[f"{col}_rolling_std_3"] = (
                df.groupby("city")[col]
                .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
            )

        logger.info("Features engineered. Final shape: %s", df.shape)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # ML preparation
    # ─────────────────────────────────────────────────────────────────────────

    _CANDIDATE_FEATURES: list[str] = [
        "hour", "day_of_week", "month",
        "is_weekend", "is_rush_hour",
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "temperature", "humidity", "wind_speed",
        "pm_ratio", "no_no2_ratio",
        "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3",
        "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
        "pm2_5_rolling_mean_3", "pm2_5_rolling_std_3",
        "aqi_rolling_mean_3",
        "city_encoded",
    ]

    def prepare_ml_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Encodes city, selects available features, drops NaN rows, and returns
        (X, y) ready for sklearn.

        Side-effect: stores `self.feature_columns` for use in the predictor.
        """
        df = df.copy()
        df["city_encoded"] = self.label_encoder.fit_transform(df["city"])

        available = [c for c in self._CANDIDATE_FEATURES if c in df.columns]
        df_ml = df[available + ["aqi"]].dropna()

        X = df_ml[available]
        y = df_ml["aqi"]

        self.feature_columns = available
        logger.info(
            "ML dataset: %d samples × %d features", X.shape[0], X.shape[1]
        )
        return X, y

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def run_pipeline(self, db_path: str) -> pd.DataFrame:
        """Load → clean → feature-engineer and return the processed DataFrame."""
        df = self.load_from_db(db_path)
        df = self.clean(df)
        df = self.engineer_features(df)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save_processed(self, df: pd.DataFrame, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Processed data saved → %s (%d rows)", out_path, len(df))
