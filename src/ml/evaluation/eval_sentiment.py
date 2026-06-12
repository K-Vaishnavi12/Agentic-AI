"""Sentiment evaluation summary."""
from __future__ import annotations

from src.ml.config import sentiment_model_dir
from src.ml.utils.io import read_json


def collect_sentiment_metrics() -> dict:
    card = read_json(sentiment_model_dir() / "model_card.json", default=None)
    if not card:
        return {"trained": False}
    return {
        "trained": True,
        "labels": card.get("labels"),
        "best": card.get("best"),
        "results": card.get("results"),
    }
