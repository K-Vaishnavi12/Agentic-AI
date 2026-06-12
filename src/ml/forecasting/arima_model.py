"""ARIMA forecaster — classical baseline using ``statsmodels``.

We fit ARIMA on the **training** close-price series and produce a one-step-
ahead forecast for each test timestamp via walk-forward refitting.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.ml.config import ForecastConfig, forecast_model_dir
from src.ml.utils.io import save_joblib, load_joblib


@dataclass
class ARIMAArtifacts:
    order: tuple[int, int, int]
    last_train_value: float
    train_size: int


class ARIMAForecaster:
    """Lightweight wrapper around ``statsmodels.tsa.arima.model.ARIMA``."""

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)):
        self.order = order
        self._fitted = None  # populated after fit

    # ── Training ────────────────────────────────────────────────
    def fit(self, train_series: Sequence[float]) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(np.asarray(train_series, dtype=float), order=self.order)
            self._fitted = model.fit()
        return self

    # ── Walk-forward backtest on the test set ───────────────────
    def walk_forward_predict(self, train_series: Sequence[float],
                              test_series: Sequence[float]) -> np.ndarray:
        """Refit at each step (cheap for ARIMA on ~1k samples)."""
        from statsmodels.tsa.arima.model import ARIMA
        history = list(map(float, train_series))
        preds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for actual in test_series:
                model = ARIMA(np.asarray(history, dtype=float), order=self.order)
                fit = model.fit()
                yhat = float(fit.forecast(steps=1)[0])
                preds.append(yhat)
                history.append(float(actual))
        return np.asarray(preds)

    # ── Inference: future forecast (no actuals) ─────────────────
    def forecast(self, steps: int = 5) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() first.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._fitted.forecast(steps=steps), dtype=float)

    # ── Persistence ─────────────────────────────────────────────
    def save(self, symbol: str) -> Path:
        if self._fitted is None:
            raise RuntimeError("Nothing to save — fit first.")
        out_dir = forecast_model_dir(symbol)
        path = out_dir / "arima.joblib"
        save_joblib({"fitted": self._fitted, "order": self.order}, path)
        return path

    @classmethod
    def load(cls, symbol: str) -> "ARIMAForecaster":
        path = forecast_model_dir(symbol) / "arima.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        blob = load_joblib(path)
        inst = cls(order=blob["order"])
        inst._fitted = blob["fitted"]
        return inst


def fit_arima_default(train: pd.Series, cfg: ForecastConfig) -> ARIMAForecaster:
    return ARIMAForecaster(order=cfg.arima_order).fit(train.values)
