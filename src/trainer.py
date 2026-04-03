"""
trainer.py — Machine Learning Pipeline
=======================================
Trains multiple regressors, picks the best by RMSE, optionally tunes
hyperparameters, and persists artefacts (model, scaler, metadata) to disk.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train, evaluate, tune, and save the best AQI regression model.

    Workflow
    --------
    1. ``train_and_select(X, y)``  →  runs all models, picks winner
    2. Optionally calls RandomizedSearchCV on the winning model family
    3. Saves ``best_model.pkl``, ``scaler.pkl``, and ``metadata.json``
    """

    def __init__(self, config: dict) -> None:
        self.config       = config
        self.save_path    = config["models"]["save_path"]
        self.test_size    = config["models"].get("test_size", 0.2)
        self.cv_folds     = config["models"].get("cv_folds", 3)
        self.n_iter       = config["models"].get("n_iter_search", 10)
        self.random_state = config["models"].get("random_state", 42)

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        # Populated after training
        self.best_model:      Optional[object]     = None
        self.best_model_name: Optional[str]        = None
        self.best_metrics:    Optional[dict]       = None
        self.scaler:          Optional[StandardScaler] = None
        self.feature_columns: Optional[list[str]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Model catalogue
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _candidate_models() -> dict:
        return {
            "LinearRegression":  LinearRegression(),
            "Ridge":             Ridge(alpha=1.0),
            "Lasso":             Lasso(alpha=0.1, max_iter=5000),
            "RandomForest":      RandomForestRegressor(
                                     n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting":  GradientBoostingRegressor(
                                     n_estimators=100, random_state=42),
            "ExtraTrees":        ExtraTreesRegressor(
                                     n_estimators=100, random_state=42, n_jobs=-1),
        }

    @staticmethod
    def _hp_grids() -> dict:
        return {
            "RandomForest": {
                "n_estimators":    [100, 200, 300],
                "max_depth":       [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
            },
            "GradientBoosting": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth":     [3, 5, 7],
                "subsample":     [0.7, 0.8, 1.0],
            },
            "ExtraTrees": {
                "n_estimators":    [100, 200, 300],
                "max_depth":       [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics(model, X_test: np.ndarray, y_test: pd.Series) -> dict:
        y_pred = model.predict(X_test)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae    = float(mean_absolute_error(y_test, y_pred))
        r2     = float(r2_score(y_test, y_pred))
        return {
            "rmse": round(rmse, 4),
            "mae":  round(mae,  4),
            "r2":   round(r2,   4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def train_and_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Train all candidate models, select the best, optionally tune it, save
        artefacts, and return a results dict keyed by model name.

        Parameters
        ----------
        X : feature matrix
        y : target (AQI)
        feature_columns : list of feature names (for metadata / predictor)

        Returns
        -------
        dict mapping model_name → {rmse, mae, r2}
        """
        self.feature_columns = feature_columns or list(X.columns)
        logger.info("ML pipeline starting — %d samples, %d features",
                    len(X), X.shape[1])

        # ── Train / test split ─────────────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # ── Scaling ────────────────────────────────────────────────────────
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        # ── Train all candidates ───────────────────────────────────────────
        results: dict = {}
        for name, model in self._candidate_models().items():
            try:
                logger.info("Training %-20s …", name)
                model.fit(X_tr_s, y_tr)
                m = self._metrics(model, X_te_s, y_te)

                # Cross-validation score on training set
                cv_rmse = float(-cross_val_score(
                    model, X_tr_s, y_tr,
                    cv=self.cv_folds,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                ).mean())
                m["cv_rmse"] = round(cv_rmse, 4)

                results[name] = {"model": model, "metrics": m}
                logger.info(
                    "  %-20s RMSE=%7.3f  MAE=%7.3f  R²=%6.4f  CV-RMSE=%7.3f",
                    name, m["rmse"], m["mae"], m["r2"], m["cv_rmse"]
                )
            except Exception as exc:
                logger.error("Training failed for %s: %s", name, exc)

        if not results:
            raise RuntimeError("All model training attempts failed.")

        # ── Select best by RMSE ────────────────────────────────────────────
        best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
        best      = results[best_name]
        logger.info("\n★ Best model (pre-tuning): %s  (RMSE=%.4f)",
                    best_name, best["metrics"]["rmse"])

        # ── Hyperparameter tuning ──────────────────────────────────────────
        hp_grids = self._hp_grids()
        if best_name in hp_grids:
            logger.info("Hyperparameter tuning for %s …", best_name)
            try:
                search = RandomizedSearchCV(
                    estimator   = best["model"].__class__(random_state=self.random_state),
                    param_distributions = hp_grids[best_name],
                    n_iter      = self.n_iter,
                    cv          = self.cv_folds,
                    scoring     = "neg_root_mean_squared_error",
                    random_state= self.random_state,
                    n_jobs      = -1,
                    verbose     = 0,
                )
                search.fit(X_tr_s, y_tr)
                tuned_metrics = self._metrics(search.best_estimator_, X_te_s, y_te)
                logger.info("Tuned RMSE=%.4f (was %.4f)",
                            tuned_metrics["rmse"], best["metrics"]["rmse"])
                if tuned_metrics["rmse"] <= best["metrics"]["rmse"]:
                    best = {"model": search.best_estimator_, "metrics": tuned_metrics}
                    logger.info("Using tuned model.")
            except Exception as exc:
                logger.warning("Hyperparameter tuning failed: %s", exc)

        # ── Persist ────────────────────────────────────────────────────────
        self.best_model      = best["model"]
        self.best_model_name = best_name
        self.best_metrics    = best["metrics"]
        self._save_artefacts(results)

        return {name: r["metrics"] for name, r in results.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_artefacts(self, all_results: dict) -> None:
        """Pickle model + scaler; write metadata JSON."""
        sp = self.save_path

        with open(f"{sp}/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        with open(f"{sp}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Feature importances (tree-based models)
        fi: Optional[dict] = None
        if hasattr(self.best_model, "feature_importances_") and self.feature_columns:
            fi = dict(zip(
                self.feature_columns,
                self.best_model.feature_importances_.tolist(),
            ))

        metadata = {
            "best_model":       self.best_model_name,
            "metrics":          self.best_metrics,
            "feature_columns":  self.feature_columns,
            "feature_importances": fi,
            "trained_at":       datetime.now().isoformat(),
            "all_results": {
                name: r["metrics"] for name, r in all_results.items()
            },
        }
        with open(f"{sp}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(
            "Artefacts saved → %s  [best=%s  RMSE=%.4f  R²=%.4f]",
            sp, self.best_model_name,
            self.best_metrics["rmse"], self.best_metrics["r2"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────────────

    def print_leaderboard(self, results: dict[str, dict]) -> None:
        """Pretty-print a model comparison table."""
        rows = sorted(results.items(), key=lambda x: x[1]["rmse"])
        header = f"\n{'Model':<24} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
        print(header)
        print("─" * len(header))
        for name, m in rows:
            star = " ★" if name == self.best_model_name else ""
            print(f"{name:<24} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}{star}")
