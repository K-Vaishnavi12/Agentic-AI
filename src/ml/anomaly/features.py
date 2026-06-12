"""Feature engineering for anomaly detection: returns, rolling stats."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given OHLCV DataFrame indexed by date, return a feature matrix.

    Columns produced:
        ret_1d, ret_5d, ret_20d
        log_vol_z (z-scored log volume)
        roll_vol_5, roll_vol_20  (rolling stdev of returns)
        hl_range (high-low range / close)
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["roll_vol_5"] = out["ret_1d"].rolling(5).std()
    out["roll_vol_20"] = out["ret_1d"].rolling(20).std()

    if "volume" in df.columns:
        log_vol = np.log1p(df["volume"].astype(float).clip(lower=0))
        mean, std = log_vol.mean(), log_vol.std() or 1e-9
        out["log_vol_z"] = (log_vol - mean) / std
    else:
        out["log_vol_z"] = 0.0

    if {"high", "low"}.issubset(df.columns):
        out["hl_range"] = (df["high"].astype(float) - df["low"].astype(float)) / close.replace(0, np.nan)
    else:
        out["hl_range"] = 0.0

    return out.dropna()


FEATURE_COLS = ["ret_1d", "ret_5d", "ret_20d", "roll_vol_5", "roll_vol_20",
                "log_vol_z", "hl_range"]
