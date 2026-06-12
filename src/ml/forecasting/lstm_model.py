"""LSTM forecaster — sliding-window regression on close-price returns.

Architecture: 2-layer LSTM -> dropout -> linear head. Trains on log-returns
of the close price (stationary, easier to learn) and reconstructs prices at
inference. We also keep an early-stopping loop on the validation MAE.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.ml.config import ForecastConfig, forecast_model_dir
from src.ml.utils.io import save_torch, load_torch, write_json, read_json
from src.ml.utils.seed import set_seed


# ── Sliding window dataset ──────────────────────────────────────────
def _make_windows(series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X[i] = series[i:i+lookback], y[i] = series[i+lookback]."""
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ── Scaler (z-score on returns) ─────────────────────────────────────
@dataclass
class ZScaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-9)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-9) + self.mean

    def to_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "ZScaler":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


# ── PyTorch model ───────────────────────────────────────────────────
def _build_model(input_size: int, hidden: int, layers: int, dropout: float):
    import torch.nn as nn

    class LSTMReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            # Last timestep
            return self.head(out[:, -1, :]).squeeze(-1)

    return LSTMReg()


# ── Forecaster ──────────────────────────────────────────────────────
class LSTMForecaster:
    """Train on log-returns; predict next-day close price."""

    def __init__(self, cfg: ForecastConfig | None = None):
        self.cfg = cfg or ForecastConfig()
        self.model = None
        self.scaler: ZScaler | None = None
        self.history: dict | None = None

    # ── Training loop ───────────────────────────────────────────
    def fit(self, train_series: pd.Series, val_series: pd.Series | None = None,
            verbose: bool = False) -> "LSTMForecaster":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        set_seed(42)
        cfg = self.cfg

        # Convert prices -> log returns; fit scaler on train returns only
        train_ret = np.diff(np.log(train_series.values.astype(float)))
        self.scaler = ZScaler(mean=float(train_ret.mean()), std=float(train_ret.std()))
        train_scaled = self.scaler.transform(train_ret)

        X_tr, y_tr = _make_windows(train_scaled, cfg.lookback)
        if len(X_tr) == 0:
            raise ValueError(
                f"Not enough training data: need >{cfg.lookback} samples, "
                f"got {len(train_series)}"
            )

        # Optional validation
        val_loader = None
        if val_series is not None and len(val_series) > cfg.lookback + 1:
            val_ret = np.diff(np.log(val_series.values.astype(float)))
            val_scaled = self.scaler.transform(val_ret)
            X_va, y_va = _make_windows(val_scaled, cfg.lookback)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_va).unsqueeze(-1),
                              torch.from_numpy(y_va)),
                batch_size=cfg.batch_size, shuffle=False,
            )

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1),
                          torch.from_numpy(y_tr)),
            batch_size=cfg.batch_size, shuffle=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model(1, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val = float("inf")
        best_state = None
        patience, bad = 5, 0

        for epoch in range(cfg.epochs):
            self.model.train()
            ep_train = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                ep_train += float(loss.item()) * xb.size(0)
            ep_train /= len(train_loader.dataset)
            train_losses.append(ep_train)

            ep_val = float("nan")
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    s = 0.0
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        s += float(loss_fn(self.model(xb), yb).item()) * xb.size(0)
                    ep_val = s / len(val_loader.dataset)
                val_losses.append(ep_val)
                if ep_val < best_val - 1e-6:
                    best_val = ep_val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        if verbose:
                            print(f"  early stop @ epoch {epoch+1}")
                        break

            if verbose:
                print(f"  epoch {epoch+1:02d} train_mse={ep_train:.5f} val_mse={ep_val:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = {"train_loss": train_losses, "val_loss": val_losses}
        return self

    # ── Backtest on test set (auto-regressive walk-forward) ─────
    def walk_forward_predict(self, train_series: pd.Series,
                              test_series: pd.Series) -> np.ndarray:
        """Predict test prices one step at a time, feeding the *actual* previous
        return back into the window (walk-forward). This is the standard
        backtest protocol for time-series forecasting.
        """
        import torch
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() first.")

        cfg = self.cfg
        all_prices = np.concatenate([train_series.values, test_series.values]).astype(float)
        all_ret = np.diff(np.log(all_prices))
        all_scaled = self.scaler.transform(all_ret)

        device = next(self.model.parameters()).device
        self.model.eval()
        preds_price = []
        train_len = len(train_series)
        # Index `i` here counts test prices (1..len(test))
        for i in range(len(test_series)):
            # Build window of LAST `lookback` returns ending just before test step i
            end = train_len - 1 + i  # last return index BEFORE the next price
            start = end - cfg.lookback
            if start < 0:
                preds_price.append(float(all_prices[train_len + i]))  # fallback
                continue
            window = all_scaled[start:end]
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                pred_scaled = float(self.model(x).cpu().numpy().ravel()[0])
            pred_ret = self.scaler.inverse(np.array([pred_scaled]))[0]
            prev_price = float(all_prices[train_len + i - 1]) if i > 0 else float(train_series.values[-1])
            preds_price.append(prev_price * float(np.exp(pred_ret)))
        return np.asarray(preds_price, dtype=float)

    # ── Forecast N future steps (no actuals) ────────────────────
    def forecast(self, recent_series: pd.Series, steps: int = 5) -> np.ndarray:
        import torch
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() first.")
        cfg = self.cfg
        if len(recent_series) < cfg.lookback + 1:
            raise ValueError(f"Need at least {cfg.lookback + 1} prices, got {len(recent_series)}")

        device = next(self.model.parameters()).device
        self.model.eval()
        prices = list(map(float, recent_series.values))
        rets = list(np.diff(np.log(np.asarray(prices))))
        rets_scaled = list(self.scaler.transform(np.asarray(rets, dtype=float)))

        out = []
        for _ in range(steps):
            window = np.asarray(rets_scaled[-cfg.lookback:], dtype=np.float32)
            x = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                pred_scaled = float(self.model(x).cpu().numpy().ravel()[0])
            rets_scaled.append(pred_scaled)
            pred_ret = float(self.scaler.inverse(np.array([pred_scaled]))[0])
            next_price = prices[-1] * float(np.exp(pred_ret))
            prices.append(next_price)
            out.append(next_price)
        return np.asarray(out, dtype=float)

    # ── Persistence ─────────────────────────────────────────────
    def save(self, symbol: str) -> Path:
        import torch
        if self.model is None or self.scaler is None:
            raise RuntimeError("Nothing to save.")
        out_dir = forecast_model_dir(symbol)
        save_torch(self.model.state_dict(), out_dir / "lstm.pt")
        write_json(out_dir / "lstm_meta.json", {
            "scaler": self.scaler.to_dict(),
            "cfg": {
                "lookback": self.cfg.lookback,
                "hidden_size": self.cfg.hidden_size,
                "num_layers": self.cfg.num_layers,
                "dropout": self.cfg.dropout,
            },
            "history": self.history,
        })
        return out_dir / "lstm.pt"

    @classmethod
    def load(cls, symbol: str) -> "LSTMForecaster":
        import torch
        out_dir = forecast_model_dir(symbol)
        meta_path = out_dir / "lstm_meta.json"
        weights_path = out_dir / "lstm.pt"
        if not (meta_path.exists() and weights_path.exists()):
            raise FileNotFoundError(weights_path)
        meta = read_json(meta_path)
        cfg = ForecastConfig(
            lookback=int(meta["cfg"]["lookback"]),
            hidden_size=int(meta["cfg"]["hidden_size"]),
            num_layers=int(meta["cfg"]["num_layers"]),
            dropout=float(meta["cfg"]["dropout"]),
        )
        inst = cls(cfg)
        inst.scaler = ZScaler.from_dict(meta["scaler"])
        inst.model = _build_model(1, cfg.hidden_size, cfg.num_layers, cfg.dropout)
        inst.model.load_state_dict(load_torch(weights_path))
        inst.model.eval()
        inst.history = meta.get("history")
        return inst
