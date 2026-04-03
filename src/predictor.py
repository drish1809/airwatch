"""
predictor.py — AQI Prediction Module
======================================
Loads saved artefacts (model + scaler + metadata) and exposes a clean
predict() interface for the Streamlit app and any other consumer.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AQIPredictor:
    """
    Wrapper around a trained sklearn pipeline stored on disk.

    Usage
    -----
    predictor = AQIPredictor("models")
    aqi = predictor.predict({"hour": 8, "pm2_5": 45, ...})
    """

    def __init__(self, models_path: str = "models") -> None:
        self.models_path = models_path
        self.model:    Optional[object] = None
        self.scaler:   Optional[object] = None
        self.metadata: Optional[dict]  = None
        self._load()

    def _load(self) -> None:
        base = Path(self.models_path)
        model_file    = base / "best_model.pkl"
        scaler_file   = base / "scaler.pkl"
        metadata_file = base / "metadata.json"

        if not model_file.exists():
            logger.warning("No model found at %s — run train_pipeline.py first.", base)
            return

        with model_file.open("rb") as f:
            self.model = pickle.load(f)
        with scaler_file.open("rb") as f:
            self.scaler = pickle.load(f)
        with metadata_file.open("r") as f:
            self.metadata = json.load(f)

        logger.info(
            "Loaded model '%s'  RMSE=%.4f  R²=%.4f  (trained %s)",
            self.metadata.get("best_model", "unknown"),
            self.metadata.get("metrics", {}).get("rmse", 0),
            self.metadata.get("metrics", {}).get("r2", 0),
            self.metadata.get("trained_at", "?"),
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    @property
    def feature_columns(self) -> list[str]:
        if self.metadata:
            return self.metadata.get("feature_columns", [])
        return []

    @property
    def model_name(self) -> str:
        if self.metadata:
            return self.metadata.get("best_model", "Unknown")
        return "Unknown"

    @property
    def metrics(self) -> dict:
        return self.metadata.get("metrics", {}) if self.metadata else {}

    @property
    def feature_importances(self) -> dict:
        return self.metadata.get("feature_importances") or {}

    def predict(self, features: dict) -> float:
        """
        Predict AQI from a feature dictionary.

        Missing features default to 0 so the caller does not need to supply
        every column when doing manual UI predictions.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Train the model first.")

        row = {col: features.get(col, 0.0) for col in self.feature_columns}
        X   = pd.DataFrame([row])[self.feature_columns]
        Xs  = self.scaler.transform(X)
        pred = self.model.predict(Xs)[0]
        return float(max(0.0, round(pred, 2)))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict AQI for a DataFrame that already contains the feature columns.
        Missing columns are filled with 0.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded.")

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        Xs = self.scaler.transform(df[self.feature_columns])
        return np.maximum(0.0, self.model.predict(Xs))

    def get_info(self) -> dict:
        """Return human-readable model info."""
        return {
            "model":      self.model_name,
            "metrics":    self.metrics,
            "features":   self.feature_columns,
            "trained_at": self.metadata.get("trained_at") if self.metadata else None,
        }
