"""Tests for the TF-IDF baseline classifiers (skip transformer for speed)."""
import os
os.environ["ML_SKIP_TRANSFORMER"] = "1"

import numpy as np
import pandas as pd

from src.ml.data.sentiment_data import load_sentiment_dataset, train_test_split_text
from src.ml.data.legal_data import load_legal_dataset, train_test_split_legal
from src.ml.sentiment.baseline_tfidf import TfidfSentimentBaseline
from src.ml.legal_classifier.tfidf_logreg import TfidfLegalBaseline
from src.ml.utils.metrics import classification_report_dict


def test_sentiment_dataset_balanced():
    df = load_sentiment_dataset(force_synthetic=True)
    counts = df["label_id"].value_counts()
    assert len(counts) == 3, "all 3 sentiment classes should be present"
    # No class should be drastically underrepresented
    assert counts.min() / counts.max() > 0.3


def test_tfidf_sentiment_beats_random():
    df = load_sentiment_dataset(force_synthetic=True)
    train_df, test_df = train_test_split_text(df, test_size=0.2, seed=42)
    clf = TfidfSentimentBaseline().fit(
        train_df["text"].tolist(), train_df["label_id"].tolist(),
    )
    preds = clf.predict(test_df["text"].tolist())
    rep = classification_report_dict(
        test_df["label_id"].tolist(), preds, labels=["neg", "neu", "pos"],
    )
    # Random would be ~33%; we expect well above with synthetic templates
    assert rep["accuracy"] > 0.6, f"acc too low: {rep['accuracy']}"


def test_legal_dataset_four_classes():
    df = load_legal_dataset(force_regenerate=True)
    counts = df["label_id"].value_counts()
    assert len(counts) == 4


def test_tfidf_legal_beats_random():
    df = load_legal_dataset(force_regenerate=True)
    train_df, test_df = train_test_split_legal(df, test_size=0.2, seed=42)
    clf = TfidfLegalBaseline().fit(
        train_df["text"].tolist(), train_df["label_id"].tolist(),
    )
    preds = clf.predict(test_df["text"].tolist())
    rep = classification_report_dict(
        test_df["label_id"].tolist(), preds,
        labels=["compliance", "regulatory", "litigation", "governance"],
    )
    assert rep["accuracy"] > 0.7, f"acc too low: {rep['accuracy']}"
