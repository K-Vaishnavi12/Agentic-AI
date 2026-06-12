"""Lightweight metric implementations.

We deliberately avoid pulling in extra deps for the simple metrics. ROUGE
uses ``rouge_score`` (already in requirements) but is imported lazily so
non-LLM evaluation stays fast.
"""
from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np


# ── Regression / forecasting ────────────────────────────────────────
def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error (in %). Robust to zeros via eps."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Fraction of times the predicted direction (up/down) matches reality.

    Computed on consecutive differences, so it answers "did we get the sign
    right?". This is the key metric for trading-relevant forecasts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # Treat 0 (no change) as a match if both are 0
    return float(np.mean(true_dir == pred_dir))


def regression_report(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }


# ── Classification ──────────────────────────────────────────────────
def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int],
                               labels: Sequence[str] | None = None) -> dict:
    """Macro-averaged precision/recall/F1 + per-class breakdown.

    Pure-numpy implementation so it works without sklearn import overhead.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        n_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
        labels = [str(i) for i in range(n_classes)]
    n_classes = len(labels)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    per_class = {}
    f1s, precs, recs = [], [], []
    for i, lbl in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(cm[i, :].sum())
        per_class[lbl] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        precs.append(precision); recs.append(recall); f1s.append(f1)

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }


# ── ROUGE for LLM evaluation (lazy) ─────────────────────────────────
def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F-measure between two strings. Returns 0.0 on import failure."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, candidate)["rougeL"].fmeasure)


def avg_rouge_l(refs: Iterable[str], cands: Iterable[str]) -> float:
    refs, cands = list(refs), list(cands)
    if not refs:
        return 0.0
    scores = [rouge_l(r, c) for r, c in zip(refs, cands)]
    return float(np.mean(scores))
