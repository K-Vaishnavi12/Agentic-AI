"""Tests for ML utilities (metrics, seed, IO)."""
import math
import numpy as np

from src.ml.utils.metrics import (
    mae, rmse, mape, directional_accuracy, regression_report,
    classification_report_dict,
)
from src.ml.utils.seed import set_seed


def test_mae_zero_when_perfect():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # errors: [1, -1] -> RMSE = 1
    assert math.isclose(rmse([0, 0], [1, -1]), 1.0)


def test_mape_handles_zero():
    # eps fallback should keep this finite
    val = mape([0.0, 1.0], [0.1, 1.1])
    assert math.isfinite(val)


def test_directional_accuracy_perfect():
    y = [1, 2, 3, 4, 5]
    p = [1.1, 2.2, 3.1, 4.2, 5.5]
    assert directional_accuracy(y, p) == 1.0


def test_directional_accuracy_inverted():
    y = [1, 2, 3, 4]
    p = [4, 3, 2, 1]   # opposite directions
    assert directional_accuracy(y, p) == 0.0


def test_regression_report_keys():
    r = regression_report([1, 2, 3], [1.1, 1.9, 3.05])
    for k in ("mae", "rmse", "mape_pct", "directional_accuracy", "n_samples"):
        assert k in r


def test_classification_report_perfect():
    r = classification_report_dict([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2],
                                    labels=["a", "b", "c"])
    assert r["accuracy"] == 1.0
    assert math.isclose(r["macro_f1"], 1.0)


def test_classification_report_partial():
    r = classification_report_dict([0, 1, 0, 1], [0, 0, 0, 1],
                                    labels=["a", "b"])
    assert 0.0 < r["accuracy"] < 1.0
    assert "per_class" in r and set(r["per_class"]) == {"a", "b"}


def test_seed_deterministic():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)
