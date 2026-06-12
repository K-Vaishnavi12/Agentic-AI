"""Inference layer for legal classifier — auto-loads best trained model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.ml.config import LEGAL_LABELS, legal_model_dir
from src.ml.utils.io import read_json


@dataclass
class LegalPrediction:
    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]


class LegalClassifier:
    def __init__(self, model_name: str, model):
        self.model_name = model_name
        self.model = model

    @classmethod
    def load(cls) -> "LegalClassifier":
        card_path = legal_model_dir() / "model_card.json"
        card = read_json(card_path, default=None)
        if card is None:
            raise FileNotFoundError(card_path)
        best = card.get("best") or "tfidf"
        if best == "distilbert":
            from src.ml.legal_classifier.distilbert_clf import DistilBertLegalClassifier
            try:
                return cls("distilbert", DistilBertLegalClassifier.load())
            except Exception:
                pass
        from src.ml.legal_classifier.tfidf_logreg import TfidfLegalBaseline
        return cls("tfidf", TfidfLegalBaseline.load())

    def predict(self, texts: Sequence[str]) -> list[LegalPrediction]:
        probs = self.model.predict_proba(list(texts))
        out: list[LegalPrediction] = []
        for i, t in enumerate(texts):
            row = probs[i]
            label_id = int(np.argmax(row))
            out.append(LegalPrediction(
                text=t,
                label=LEGAL_LABELS[label_id],
                label_id=label_id,
                confidence=float(row[label_id]),
                probabilities={LEGAL_LABELS[j]: float(row[j]) for j in range(len(row))},
            ))
        return out
