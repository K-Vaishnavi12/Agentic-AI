"""Trainer CLI: trains ARIMA + LSTM for every cached symbol, picks the
best model on validation MAE, writes a model card per symbol.

Usage:
    python -m src.ml.forecasting.train_forecast            # all symbols
    python -m src.ml.forecasting.train_forecast SENSEX TCS # specific
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.config import ForecastConfig, forecast_model_dir, models_dir, plots_dir
from src.ml.data.stock_history import ensure_history, list_available_symbols
from src.ml.forecasting.arima_model import ARIMAForecaster
from src.ml.forecasting.lstm_model import LSTMForecaster
from src.ml.utils.io import write_json
from src.ml.utils.metrics import regression_report
from src.ml.utils.plots import plot_forecast, plot_loss_curve
from src.ml.utils.seed import set_seed


def _split(series: pd.Series, train_split: float, val_split: float):
    n = len(series)
    n_train = int(n * train_split)
    # Inside the train chunk, last `val_split` becomes validation
    n_val = max(int(n_train * val_split), 30)
    train = series.iloc[: n_train - n_val]
    val = series.iloc[n_train - n_val : n_train]
    test = series.iloc[n_train:]
    return train, val, test


def train_one(symbol: str, cfg: ForecastConfig | None = None,
              verbose: bool = True) -> dict:
    cfg = cfg or ForecastConfig()
    set_seed(42)

    df = ensure_history(symbol)
    if "close" not in df.columns:
        raise ValueError(f"{symbol}: no 'close' column in history")
    series = df["close"].dropna()
    if len(series) < cfg.lookback + 100:
        raise ValueError(f"{symbol}: too few rows ({len(series)})")

    train, val, test = _split(series, cfg.train_split, cfg.val_split)
    if verbose:
        print(f"\n[{symbol}] train={len(train)} val={len(val)} test={len(test)}")

    results: dict = {"symbol": symbol, "n_train": len(train),
                     "n_val": len(val), "n_test": len(test)}

    # ── ARIMA ──────────────────────────────────────────────────
    t0 = time.time()
    try:
        arima = ARIMAForecaster(order=cfg.arima_order)
        # Walk-forward backtest is the fair comparison; also fit final on full train+val
        arima_preds = arima.walk_forward_predict(
            train_series=pd.concat([train, val]).values,
            test_series=test.values,
        )
        arima_metrics = regression_report(test.values, arima_preds)
        # Final fit for save (uses train+val so future forecasts use everything)
        arima.fit(pd.concat([train, val]).values)
        arima.save(symbol)
        results["arima"] = {"metrics": arima_metrics,
                            "fit_seconds": round(time.time() - t0, 2)}
        if verbose:
            print(f"  ARIMA  MAE={arima_metrics['mae']:.3f} "
                  f"RMSE={arima_metrics['rmse']:.3f} "
                  f"DirAcc={arima_metrics['directional_accuracy']:.3f}")
    except Exception as e:
        results["arima"] = {"error": str(e)}
        if verbose:
            print(f"  ARIMA  failed: {e}")
        arima_preds = None
        arima_metrics = None

    # ── LSTM ───────────────────────────────────────────────────
    t0 = time.time()
    try:
        lstm = LSTMForecaster(cfg)
        lstm.fit(train, val_series=val, verbose=False)
        lstm_preds = lstm.walk_forward_predict(pd.concat([train, val]), test)
        lstm_metrics = regression_report(test.values, lstm_preds)
        lstm.save(symbol)
        results["lstm"] = {"metrics": lstm_metrics,
                           "fit_seconds": round(time.time() - t0, 2)}
        if verbose:
            print(f"  LSTM   MAE={lstm_metrics['mae']:.3f} "
                  f"RMSE={lstm_metrics['rmse']:.3f} "
                  f"DirAcc={lstm_metrics['directional_accuracy']:.3f}")

        # Loss curve plot
        if lstm.history:
            plot_loss_curve(
                lstm.history.get("train_loss") or [],
                lstm.history.get("val_loss") or [],
                plots_dir() / f"forecast_{symbol}_lstm_loss.png",
                title=f"LSTM training loss — {symbol}",
            )
    except Exception as e:
        results["lstm"] = {"error": str(e)}
        if verbose:
            print(f"  LSTM   failed: {e}")
        lstm_preds = None
        lstm_metrics = None

    # ── Pick best (by MAE) ─────────────────────────────────────
    best = None
    if arima_metrics and lstm_metrics:
        best = "lstm" if lstm_metrics["mae"] <= arima_metrics["mae"] else "arima"
    elif arima_metrics:
        best = "arima"
    elif lstm_metrics:
        best = "lstm"
    results["best_model"] = best

    # ── Plot the best model's predictions ──────────────────────
    if best == "arima" and arima_preds is not None:
        plot_forecast(test.index, test.values, arima_preds, symbol,
                      plots_dir() / f"forecast_{symbol}_arima.png",
                      title=f"ARIMA — {symbol}")
    if best == "lstm" and lstm_preds is not None:
        plot_forecast(test.index, test.values, lstm_preds, symbol,
                      plots_dir() / f"forecast_{symbol}_lstm.png",
                      title=f"LSTM — {symbol}")

    # ── Model card ─────────────────────────────────────────────
    write_json(forecast_model_dir(symbol) / "model_card.json", {
        "symbol": symbol,
        "best_model": best,
        "config": {
            "lookback": cfg.lookback,
            "horizon": cfg.horizon,
            "arima_order": list(cfg.arima_order),
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "epochs": cfg.epochs,
        },
        "results": results,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return results


def train_all(symbols: list[str] | None = None) -> dict[str, dict]:
    symbols = symbols or list_available_symbols()
    if not symbols:
        from src.ml.data.stock_history import main as ensure_main
        ensure_main()
        symbols = list_available_symbols()
    out: dict[str, dict] = {}
    for sym in symbols:
        try:
            out[sym] = train_one(sym)
        except Exception as e:
            print(f"[forecast] {sym} failed: {e}")
            out[sym] = {"error": str(e)}
    write_json(models_dir() / "forecast" / "INDEX.json", {
        "symbols": list(out.keys()),
        "best_models": {s: r.get("best_model") for s, r in out.items() if isinstance(r, dict)},
    })
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="Symbols to train (default: all cached)")
    args = ap.parse_args()
    train_all(args.symbols or None)


if __name__ == "__main__":
    _cli()
