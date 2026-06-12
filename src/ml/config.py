"""Centralised ML config — paths, hyperparameters, symbols, label spaces.

All ML modules import from here so we have a single source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.config import get_settings


# ── Symbols we forecast / monitor ────────────────────────────────────
# Yahoo Finance tickers, kept aligned with src/pipelines/stock_api.py
YF_TICKERS: List[str] = [
    "^BSESN", "^NSEI",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
]

# Display name (matches data/stock/stock.json) -> Yahoo ticker
DISPLAY_TO_YF: dict[str, str] = {
    "SENSEX": "^BSESN",
    "NIFTY50": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
}
YF_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_YF.items()}


# ── Sentiment label space ───────────────────────────────────────────
SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]
SENTIMENT_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL: dict[int, str] = {i: l for l, i in SENTIMENT_LABEL2ID.items()}


# ── Legal classifier label space ────────────────────────────────────
LEGAL_LABELS: List[str] = ["compliance", "regulatory", "litigation", "governance"]
LEGAL_LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LEGAL_LABELS)}
LEGAL_ID2LABEL: dict[int, str] = {i: l for l, i in LEGAL_LABEL2ID.items()}


# ── Hyperparameters (defaults; override via ForecastConfig etc.) ────
@dataclass
class ForecastConfig:
    lookback: int = 60          # window size fed to the LSTM
    horizon: int = 5            # how many days ahead we predict
    train_split: float = 0.8    # 80/20 chronological split
    val_split: float = 0.1      # of train -> val
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    arima_order: tuple[int, int, int] = (5, 1, 0)


@dataclass
class AnomalyConfig:
    contamination: float = 0.05         # for IsolationForest
    n_estimators: int = 200
    ae_hidden: list[int] = field(default_factory=lambda: [16, 8, 4, 8, 16])
    ae_epochs: int = 50
    ae_lr: float = 1e-3
    ae_batch_size: int = 32
    score_threshold_quantile: float = 0.95   # top 5% reconstruction error -> anomaly


@dataclass
class SentimentConfig:
    base_model: str = "distilbert-base-uncased"
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


@dataclass
class LegalConfig:
    base_model: str = "distilbert-base-uncased"
    max_len: int = 128
    batch_size: int = 8
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01


# ── Path helpers ────────────────────────────────────────────────────
def models_dir() -> Path:
    return Path(get_settings().models_dir)


def reports_dir() -> Path:
    return Path(get_settings().reports_dir)


def plots_dir() -> Path:
    p = reports_dir() / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ml_data_dir() -> Path:
    return Path(get_settings().ml_data_dir)


def forecast_model_dir(symbol: str) -> Path:
    p = models_dir() / "forecast" / symbol
    p.mkdir(parents=True, exist_ok=True)
    return p


def anomaly_model_dir() -> Path:
    p = models_dir() / "anomaly"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sentiment_model_dir() -> Path:
    p = models_dir() / "sentiment"
    p.mkdir(parents=True, exist_ok=True)
    return p


def legal_model_dir() -> Path:
    p = models_dir() / "legal"
    p.mkdir(parents=True, exist_ok=True)
    return p
