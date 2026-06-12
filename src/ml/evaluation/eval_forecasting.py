"""Re-run forecast backtest from saved model cards (no retraining)."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.ml.config import models_dir, plots_dir
from src.ml.utils.io import read_json


def collect_forecast_metrics() -> dict:
    """Aggregate every saved model_card under models/forecast/."""
    base = models_dir() / "forecast"
    out: dict = {"per_symbol": {}, "summary": {}}
    if not base.exists():
        return out
    maes_lstm, maes_arima, dirs_lstm, dirs_arima = [], [], [], []
    for sym_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        card = read_json(sym_dir / "model_card.json", default=None)
        if not card:
            continue
        results = card.get("results", {})
        sym = card.get("symbol", sym_dir.name)
        out["per_symbol"][sym] = {
            "best_model": card.get("best_model"),
            "lstm_metrics": (results.get("lstm") or {}).get("metrics"),
            "arima_metrics": (results.get("arima") or {}).get("metrics"),
        }
        m_l = (results.get("lstm") or {}).get("metrics") or {}
        m_a = (results.get("arima") or {}).get("metrics") or {}
        if "mae" in m_l: maes_lstm.append(m_l["mae"])
        if "mae" in m_a: maes_arima.append(m_a["mae"])
        if "directional_accuracy" in m_l: dirs_lstm.append(m_l["directional_accuracy"])
        if "directional_accuracy" in m_a: dirs_arima.append(m_a["directional_accuracy"])

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else None
    out["summary"] = {
        "lstm_mean_mae": _mean(maes_lstm),
        "arima_mean_mae": _mean(maes_arima),
        "lstm_mean_dir_acc": _mean(dirs_lstm),
        "arima_mean_dir_acc": _mean(dirs_arima),
        "n_symbols_evaluated": len(out["per_symbol"]),
    }
    return out
