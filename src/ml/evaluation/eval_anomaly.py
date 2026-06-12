"""Anomaly evaluation summary (reads saved model card)."""
from __future__ import annotations

from src.ml.config import anomaly_model_dir
from src.ml.utils.io import read_json


def collect_anomaly_metrics() -> dict:
    card = read_json(anomaly_model_dir() / "model_card.json", default=None)
    if not card:
        return {"trained": False}
    return {
        "trained": True,
        "best": card.get("best"),
        "results": card.get("results"),
        "n_train": card.get("n_train"),
        "feature_cols": card.get("feature_cols"),
    }
