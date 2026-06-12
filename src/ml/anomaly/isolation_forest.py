"""Isolation Forest anomaly detector — sklearn baseline."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.config import AnomalyConfig, anomaly_model_dir
from src.ml.utils.io import save_joblib, load_joblib


class IsoForestDetector:
    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self.model = None
        self._feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "IsoForestDetector":
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self._feature_cols = list(X.columns)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.values)
        self.model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            contamination=self.cfg.contamination,
            random_state=42,
        ).fit(Xs)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (we negate sklearn's decision_function)."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        Xs = self._scaler.transform(X[self._feature_cols].values)
        # decision_function: higher = more normal. Flip sign.
        return -np.asarray(self.model.decision_function(Xs))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = anomaly, 0 = normal (matches our project-wide convention)."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        Xs = self._scaler.transform(X[self._feature_cols].values)
        raw = self.model.predict(Xs)   # sklearn: -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def save(self) -> Path:
        path = anomaly_model_dir() / "isolation_forest.joblib"
        save_joblib({
            "model": self.model,
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
            "cfg": vars(self.cfg),
        }, path)
        return path

    @classmethod
    def load(cls) -> "IsoForestDetector":
        path = anomaly_model_dir() / "isolation_forest.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        blob = load_joblib(path)
        inst = cls(AnomalyConfig(**blob["cfg"]))
        inst.model = blob["model"]
        inst._scaler = blob["scaler"]
        inst._feature_cols = blob["feature_cols"]
        return inst
