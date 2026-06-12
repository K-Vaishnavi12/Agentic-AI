"""Train script for anomaly detectors (IsolationForest + Autoencoder).

Trains on **all available stock symbols pooled together** so the detector
generalises across stocks (the features are scale-invariant returns/ratios).

Evaluation: we inject synthetic anomalies (huge return spikes, volume shocks)
into a held-out window and measure ROC-AUC + Precision@k.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.ml.anomaly.autoencoder import AutoEncoderDetector
from src.ml.anomaly.features import build_features, FEATURE_COLS
from src.ml.anomaly.isolation_forest import IsoForestDetector
from src.ml.config import AnomalyConfig, anomaly_model_dir, plots_dir
from src.ml.data.stock_history import ensure_history, list_available_symbols
from src.ml.utils.io import write_json
from src.ml.utils.plots import plot_anomaly, plot_loss_curve
from src.ml.utils.seed import set_seed


def _pool_features(symbols: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = ensure_history(sym)
        except Exception:
            continue
        feats = build_features(df)
        feats["__symbol__"] = sym
        frames.append(feats)
    if not frames:
        raise RuntimeError("No symbol histories available for anomaly training.")
    pooled = pd.concat(frames, axis=0)
    return pooled


def _inject_synthetic_anomalies(X: pd.DataFrame, rate: float = 0.02,
                                  seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a copy of X with `rate` rows perturbed; y=1 marks anomalies.

    Perturbations are large positive/negative shocks to ret_1d and a big
    spike in log_vol_z — exactly what we'd want a detector to flag.
    """
    rng = np.random.default_rng(seed)
    Xc = X.copy().reset_index(drop=True)
    y = np.zeros(len(Xc), dtype=int)
    n_anom = max(1, int(len(Xc) * rate))
    idx = rng.choice(len(Xc), size=n_anom, replace=False)
    for i in idx:
        sign = rng.choice([-1, 1])
        Xc.loc[i, "ret_1d"] = float(Xc.loc[i, "ret_1d"]) + sign * 0.10
        Xc.loc[i, "ret_5d"] = float(Xc.loc[i, "ret_5d"]) + sign * 0.12
        Xc.loc[i, "log_vol_z"] = float(Xc.loc[i, "log_vol_z"]) + 4.0
        Xc.loc[i, "roll_vol_5"] = float(Xc.loc[i, "roll_vol_5"]) + 0.05
        y[i] = 1
    return Xc, y


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pure-numpy ROC-AUC (Mann-Whitney U). Avoids sklearn import overhead."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # Pairwise wins + 0.5 * ties, vectorised
    diff = pos_scores[:, None] - neg_scores[None, :]
    wins = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return wins / (len(pos_scores) * len(neg_scores))


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    top = y_true[order[:k]]
    return float(top.mean()) if len(top) else float("nan")


def train_all(verbose: bool = True) -> dict:
    set_seed(42)
    symbols = list_available_symbols()
    if not symbols:
        from src.ml.data.stock_history import main as ensure_main
        ensure_main()
        symbols = list_available_symbols()

    pooled = _pool_features(symbols)
    X = pooled[FEATURE_COLS].copy()
    if verbose:
        print(f"[anomaly] pooled rows: {len(X)} from {len(symbols)} symbols")

    # ── Hold-out for evaluation: inject synthetic anomalies ──────
    n_test = max(int(len(X) * 0.2), 100)
    X_train = X.iloc[:-n_test].copy()
    X_test_raw = X.iloc[-n_test:].copy()
    X_test, y_test = _inject_synthetic_anomalies(X_test_raw, rate=0.05)

    cfg = AnomalyConfig()
    results: dict = {}

    # ── IsolationForest ────────────────────────────────────────
    iso = IsoForestDetector(cfg).fit(X_train)
    iso.save()
    iso_scores = iso.score(X_test)
    results["isolation_forest"] = {
        "roc_auc": _roc_auc(y_test, iso_scores),
        "precision_at_5pct": _precision_at_k(y_test, iso_scores, max(1, int(len(X_test) * 0.05))),
        "n_test": int(len(X_test)),
        "n_anomalies_injected": int(y_test.sum()),
    }
    if verbose:
        print(f"  IsolationForest ROC-AUC={results['isolation_forest']['roc_auc']:.3f} "
              f"P@5%={results['isolation_forest']['precision_at_5pct']:.3f}")
    plot_anomaly(iso_scores, threshold=float(np.quantile(iso_scores, 0.95)),
                 out_path=plots_dir() / "anomaly_iso_scores.png",
                 title="IsolationForest scores (test window, synthetic anomalies)")

    # ── Autoencoder ────────────────────────────────────────────
    ae = AutoEncoderDetector(cfg).fit(X_train, verbose=False)
    ae.save()
    ae_scores = ae.score(X_test)
    results["autoencoder"] = {
        "roc_auc": _roc_auc(y_test, ae_scores),
        "precision_at_5pct": _precision_at_k(y_test, ae_scores, max(1, int(len(X_test) * 0.05))),
        "threshold": ae.threshold,
        "n_test": int(len(X_test)),
    }
    if verbose:
        print(f"  AutoEncoder    ROC-AUC={results['autoencoder']['roc_auc']:.3f} "
              f"P@5%={results['autoencoder']['precision_at_5pct']:.3f}")
    plot_anomaly(ae_scores, ae.threshold,
                 out_path=plots_dir() / "anomaly_ae_scores.png",
                 title="Autoencoder reconstruction error (test window)")
    if ae.history and ae.history.get("train_loss"):
        plot_loss_curve(ae.history["train_loss"], None,
                        out_path=plots_dir() / "anomaly_ae_loss.png",
                        title="Autoencoder training loss")

    # ── Pick best by ROC-AUC ───────────────────────────────────
    best = max(results, key=lambda k: results[k].get("roc_auc", -1))
    write_json(anomaly_model_dir() / "model_card.json", {
        "task": "anomaly_detection",
        "models": list(results.keys()),
        "best": best,
        "results": results,
        "config": vars(cfg),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_train": int(len(X_train)),
        "feature_cols": FEATURE_COLS,
    })
    return results


if __name__ == "__main__":
    train_all()
