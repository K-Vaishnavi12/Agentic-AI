"""TF-IDF + LogisticRegression baseline for sentiment classification.

Cheap, fast, surprisingly competitive on small text classification tasks.
We always train it as a baseline / fallback for when the transformer is
disabled (CPU-only laptop, etc.).
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.ml.config import sentiment_model_dir, SENTIMENT_LABELS
from src.ml.utils.io import save_joblib, load_joblib


class TfidfSentimentBaseline:
    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfSentimentBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=None,
            random_state=42,
        ).fit(X, np.asarray(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)

    def save(self) -> Path:
        path = sentiment_model_dir() / "tfidf_baseline.joblib"
        save_joblib({"vectorizer": self.vectorizer, "clf": self.clf}, path)
        return path

    @classmethod
    def load(cls) -> "TfidfSentimentBaseline":
        path = sentiment_model_dir() / "tfidf_baseline.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        blob = load_joblib(path)
        inst = cls()
        inst.vectorizer = blob["vectorizer"]
        inst.clf = blob["clf"]
        return inst
