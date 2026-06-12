"""Tests for the anomaly detection module (synthetic, fast)."""
import numpy as np
import pandas as pd
import pytest

from src.ml.anomaly.features import build_features, FEATURE_COLS
from src.ml.anomaly.isolation_forest import IsoForestDetector
from src.ml.anomaly.autoencoder import AutoEncoderDetector
from src.ml.config import AnomalyConfig
from src.ml.data.stock_history import synthesize_history


def test_build_features_shape():
    df = synthesize_history("TEST", n_days=120)
    X = build_features(df)
    assert set(FEATURE_COLS).issubset(X.columns)
    # Rolling 20d => first ~20 rows dropped
    assert len(X) <= len(df) and len(X) >= len(df) - 20


def test_iso_forest_fit_predict():
    df = synthesize_history("TEST_IF", n_days=300)
    X = build_features(df)[FEATURE_COLS]
    det = IsoForestDetector(AnomalyConfig(contamination=0.05, n_estimators=50)).fit(X)
    preds = det.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})
    # contamination=5% means ~5% predicted anomalies
    rate = preds.mean()
    assert 0.01 <= rate <= 0.15


def test_autoencoder_fit_predict_quick():
    df = synthesize_history("TEST_AE", n_days=300)
    X = build_features(df)[FEATURE_COLS]
    det = AutoEncoderDetector(AnomalyConfig(ae_hidden=[8, 4, 8],
                                              ae_epochs=3,
                                              ae_batch_size=16)).fit(X, verbose=False)
    scores = det.score(X)
    preds = det.predict(X)
    assert scores.shape == (len(X),)
    assert preds.shape == (len(X),)
    # Threshold is set at 95th percentile -> at most ~5% flagged
    assert preds.mean() <= 0.10
