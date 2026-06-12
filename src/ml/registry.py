"""Single entry point for loading any trained model. Keeps the agents
ignorant of where artifacts live and which framework produced them.

Each loader is cached so concurrent requests reuse the same in-memory model.
"""
from __future__ import annotations
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.ml.config import (
    models_dir,
    forecast_model_dir,
    anomaly_model_dir,
    sentiment_model_dir,
    legal_model_dir,
)


_lock = threading.Lock()


# ── Forecasting ─────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def get_forecaster(symbol: str):
    """Return ``Forecaster`` for the symbol, or None if not trained."""
    from src.ml.forecasting.predict import Forecaster
    try:
        return Forecaster.load(symbol)
    except FileNotFoundError:
        return None


# ── Anomaly ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_anomaly_detector():
    from src.ml.anomaly.detect import AnomalyDetector
    try:
        return AnomalyDetector.load()
    except FileNotFoundError:
        return None


# ── Sentiment ───────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_sentiment_classifier():
    from src.ml.sentiment.predict import SentimentClassifier
    try:
        return SentimentClassifier.load()
    except FileNotFoundError:
        return None


# ── Legal classifier ────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_legal_classifier():
    from src.ml.legal_classifier.predict import LegalClassifier
    try:
        return LegalClassifier.load()
    except FileNotFoundError:
        return None


# ── Status snapshot for /metrics endpoint ──────────────────────────
def model_status() -> dict[str, Any]:
    """Quick check: which models are trained and ready."""
    status: dict[str, Any] = {}
    fdir = models_dir() / "forecast"
    forecast_syms: list[str] = []
    if fdir.exists():
        forecast_syms = sorted(p.name for p in fdir.iterdir()
                               if p.is_dir() and (p / "model_card.json").exists())
    status["forecast"] = {"trained_symbols": forecast_syms}
    status["anomaly"] = {"trained": (anomaly_model_dir() / "model_card.json").exists()}
    status["sentiment"] = {"trained": (sentiment_model_dir() / "model_card.json").exists()}
    status["legal"] = {"trained": (legal_model_dir() / "model_card.json").exists()}
    return status
