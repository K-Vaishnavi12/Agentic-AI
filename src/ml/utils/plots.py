"""Plot helpers — all functions are non-interactive (Agg backend) and write PNGs."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Forecasting ─────────────────────────────────────────────────────
def plot_forecast(dates, y_true, y_pred, symbol: str, out_path: str | Path,
                  title: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, y_true, label="actual", color="#1f3b32", linewidth=1.6)
    ax.plot(dates, y_pred, label="predicted", color="#c9a87c", linewidth=1.6, linestyle="--")
    ax.set_title(title or f"Forecast vs Actual — {symbol}")
    ax.set_xlabel("date"); ax.set_ylabel("close price")
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ── Anomaly ─────────────────────────────────────────────────────────
def plot_anomaly(scores: Sequence[float], threshold: float, out_path: str | Path,
                 title: str = "Anomaly scores") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    scores = np.asarray(scores, dtype=float)
    idx = np.arange(len(scores))
    ax.plot(idx, scores, color="#1f3b32", linewidth=1.0, label="reconstruction error")
    ax.axhline(threshold, color="#c75d5d", linestyle="--", label=f"threshold={threshold:.3f}")
    flagged = idx[scores >= threshold]
    if len(flagged):
        ax.scatter(flagged, scores[flagged], color="#c75d5d", s=18, label="anomaly")
    ax.set_title(title); ax.set_xlabel("sample"); ax.set_ylabel("score")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ── Confusion matrix ────────────────────────────────────────────────
def plot_confusion_matrix(cm, labels, out_path: str | Path,
                          title: str = "Confusion matrix") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    # Annotate cells
    thr = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ── Loss curves ─────────────────────────────────────────────────────
def plot_loss_curve(train: Sequence[float], val: Sequence[float] | None,
                    out_path: str | Path, title: str = "Loss") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train) + 1), train, label="train", color="#1f3b32")
    if val is not None and len(val):
        ax.plot(range(1, len(val) + 1), val, label="val", color="#c9a87c")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
