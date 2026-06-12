"""Tests for forecasting (synthetic data so they run fast & offline)."""
import numpy as np
import pandas as pd
import pytest

from src.ml.config import ForecastConfig
from src.ml.data.stock_history import synthesize_history
from src.ml.forecasting.lstm_model import LSTMForecaster, _make_windows


def test_make_windows_shape():
    arr = np.arange(10, dtype=np.float32)
    X, y = _make_windows(arr, lookback=3)
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    np.testing.assert_array_equal(X[0], np.array([0, 1, 2], dtype=np.float32))
    assert y[0] == 3.0


def test_synthesize_history_shape():
    df = synthesize_history("TEST", n_days=300)
    assert len(df) == 300
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].notna().all()


@pytest.mark.parametrize("epochs", [2])
def test_lstm_fit_and_forecast(epochs):
    df = synthesize_history("TEST", n_days=300)
    series = df["close"]
    cfg = ForecastConfig(lookback=20, hidden_size=16, num_layers=1,
                         epochs=epochs, batch_size=16)
    model = LSTMForecaster(cfg)
    model.fit(series.iloc[:200], val_series=series.iloc[200:240], verbose=False)
    fc = model.forecast(series.iloc[-25:], steps=5)
    assert fc.shape == (5,)
    assert np.all(np.isfinite(fc))
    assert (fc > 0).all()


def test_lstm_walk_forward_smoke():
    df = synthesize_history("TEST2", n_days=300)
    series = df["close"]
    cfg = ForecastConfig(lookback=15, hidden_size=8, num_layers=1,
                         epochs=2, batch_size=16)
    model = LSTMForecaster(cfg).fit(series.iloc[:230], verbose=False)
    preds = model.walk_forward_predict(series.iloc[:230], series.iloc[230:280])
    assert preds.shape == (50,)
    assert np.all(np.isfinite(preds))
