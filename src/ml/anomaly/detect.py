"""Combined anomaly detector — loads both models, returns ensemble flags
and per-symbol breakdowns for the Risk Agent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.ml.anomaly.autoencoder import AutoEncoderDetector
from src.ml.anomaly.features import build_features, FEATURE_COLS
from src.ml.anomaly.isolation_forest import IsoForestDetector
from src.ml.config import anomaly_model_dir
from src.ml.data.stock_history import ensure_history
from src.ml.utils.io import read_json


@dataclass
class AnomalyHit:
    symbol: str
    date: str
    score: float
    iso_flag: bool
    ae_flag: bool
    features: dict


class AnomalyDetector:
    def __init__(self, iso: IsoForestDetector, ae: AutoEncoderDetector,
                 model_card: dict | None = None):
        self.iso = iso
        self.ae = ae
        self.model_card = model_card or {}

    @classmethod
    def load(cls) -> "AnomalyDetector":
        card = read_json(anomaly_model_dir() / "model_card.json", default=None)
        if card is None:
            raise FileNotFoundError(anomaly_model_dir() / "model_card.json")
        iso = IsoForestDetector.load()
        ae = AutoEncoderDetector.load()
        return cls(iso, ae, card)

    def detect(self, symbols: Iterable[str], lookback_days: int = 30,
               top_k: int = 10) -> list[AnomalyHit]:
        """Score the most recent ``lookback_days`` for each symbol, return
        the union of anomalies sorted by combined score (descending).
        """
        hits: list[AnomalyHit] = []
        for sym in symbols:
            try:
                df = ensure_history(sym)
            except Exception:
                continue
            feats = build_features(df).iloc[-lookback_days:]
            if feats.empty:
                continue
            iso_flag = self.iso.predict(feats[FEATURE_COLS]).astype(bool)
            ae_score = self.ae.score(feats[FEATURE_COLS])
            ae_flag = (ae_score >= self.ae.threshold).astype(bool)
            iso_score = self.iso.score(feats[FEATURE_COLS])
            # Combined score: z-normalised iso + ae
            iso_z = (iso_score - iso_score.mean()) / (iso_score.std() + 1e-9)
            ae_z = (ae_score - ae_score.mean()) / (ae_score.std() + 1e-9)
            combined = (iso_z + ae_z) / 2.0
            for i in range(len(feats)):
                if iso_flag[i] or ae_flag[i]:
                    row = feats.iloc[i]
                    hits.append(AnomalyHit(
                        symbol=sym,
                        date=str(feats.index[i].date()) if hasattr(feats.index[i], "date") else str(feats.index[i]),
                        score=float(combined[i]),
                        iso_flag=bool(iso_flag[i]),
                        ae_flag=bool(ae_flag[i]),
                        features={c: float(row[c]) for c in FEATURE_COLS},
                    ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
