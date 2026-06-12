"""Inference layer: a single ``Forecaster`` class that loads whichever model
was selected as best for a given symbol and produces N-day forecasts.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.config import forecast_model_dir
from src.ml.data.stock_history import ensure_history
from src.ml.utils.io import read_json


@dataclass
class ForecastResult:
    symbol: str
    horizon: int
    predictions: list[float]            # next-N close prices
    pct_changes: list[float]            # day-over-day % changes
    last_close: float
    direction: str                      # "up" / "down" / "flat"
    model: str                          # "arima" or "lstm"
    metrics: dict | None                # eval metrics from training
    as_of: str | None                   # ISO timestamp of training


class Forecaster:
    """Symbol-specific forecaster that loads the best trained model."""

    def __init__(self, symbol: str, model_name: str, model):
        self.symbol = symbol
        self.model_name = model_name
        self.model = model
        self._card: dict | None = None

    @classmethod
    def load(cls, symbol: str) -> "Forecaster":
        card_path = forecast_model_dir(symbol) / "model_card.json"
        if not card_path.exists():
            raise FileNotFoundError(card_path)
        card = read_json(card_path)
        best = card.get("best_model")
        if best == "lstm":
            from src.ml.forecasting.lstm_model import LSTMForecaster
            model = LSTMForecaster.load(symbol)
        elif best == "arima":
            from src.ml.forecasting.arima_model import ARIMAForecaster
            model = ARIMAForecaster.load(symbol)
        else:
            raise FileNotFoundError(f"No usable model for {symbol}")
        inst = cls(symbol, best, model)
        inst._card = card
        return inst

    def predict(self, steps: int = 5) -> ForecastResult:
        if self.model_name == "lstm":
            df = ensure_history(self.symbol)
            series = df["close"].dropna()
            preds = self.model.forecast(series.iloc[-(self.model.cfg.lookback + 5):], steps=steps)
            last_close = float(series.iloc[-1])
        else:  # arima — fitted model already holds full history
            preds = self.model.forecast(steps=steps)
            df = ensure_history(self.symbol)
            last_close = float(df["close"].dropna().iloc[-1])

        preds_list = [float(p) for p in preds]
        chained = [last_close] + preds_list
        pct = [round((chained[i + 1] / chained[i] - 1.0) * 100.0, 3)
               for i in range(len(preds_list))]

        delta = preds_list[-1] - last_close
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")

        metrics = None
        as_of = None
        if self._card and "results" in self._card:
            results = self._card["results"]
            mblock = results.get(self.model_name) or {}
            metrics = mblock.get("metrics")
            as_of = self._card.get("trained_at")

        return ForecastResult(
            symbol=self.symbol,
            horizon=steps,
            predictions=preds_list,
            pct_changes=pct,
            last_close=last_close,
            direction=direction,
            model=self.model_name,
            metrics=metrics,
            as_of=as_of,
        )


def predict(symbol: str, steps: int = 5) -> Optional[ForecastResult]:
    """Convenience: returns None if no model is trained for the symbol."""
    try:
        return Forecaster.load(symbol).predict(steps=steps)
    except FileNotFoundError:
        return None
