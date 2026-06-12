"""Sentiment inference layer — auto-loads whichever model the trainer picked."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.ml.config import SENTIMENT_LABELS, sentiment_model_dir
from src.ml.utils.io import read_json


@dataclass
class SentimentPrediction:
    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]


class SentimentClassifier:
    def __init__(self, model_name: str, model):
        self.model_name = model_name
        self.model = model

    @classmethod
    def load(cls) -> "SentimentClassifier":
        card_path = sentiment_model_dir() / "model_card.json"
        card = read_json(card_path, default=None)
        if card is None:
            raise FileNotFoundError(card_path)
        best = card.get("best") or "tfidf"
        if best == "distilbert":
            from src.ml.sentiment.finbert_classifier import FinSentimentClassifier
            try:
                return cls("distilbert", FinSentimentClassifier.load())
            except Exception:
                # Fallback to TF-IDF if transformer artifacts missing/corrupt
                pass
        from src.ml.sentiment.baseline_tfidf import TfidfSentimentBaseline
        return cls("tfidf", TfidfSentimentBaseline.load())

    def predict(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        if self.model_name == "tfidf":
            probs = self.model.predict_proba(list(texts))
        else:
            probs = self.model.predict_proba(list(texts))
        out: list[SentimentPrediction] = []
        for i, t in enumerate(texts):
            row = probs[i]
            label_id = int(np.argmax(row))
            label = SENTIMENT_LABELS[label_id]
            out.append(SentimentPrediction(
                text=t, label=label, label_id=label_id,
                confidence=float(row[label_id]),
                probabilities={SENTIMENT_LABELS[j]: float(row[j]) for j in range(len(row))},
            ))
        return out
