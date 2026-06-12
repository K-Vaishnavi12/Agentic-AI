"""Dense autoencoder anomaly detector — learns to reconstruct normal points;
high reconstruction error -> anomaly.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.config import AnomalyConfig, anomaly_model_dir
from src.ml.utils.io import save_torch, load_torch, write_json, read_json
from src.ml.utils.seed import set_seed


def _build_ae(in_dim: int, hidden: list[int]):
    import torch.nn as nn

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = in_dim
            mid = len(hidden) // 2
            # Encoder
            for h in hidden[:mid + 1]:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            # Decoder (drop last ReLU at the end)
            for h in hidden[mid + 1:]:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, in_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return AutoEncoder()


class AutoEncoderDetector:
    """Reconstruction-error based anomaly detector.

    Threshold defaults to the configured quantile (e.g. 95th) of the training
    error distribution.
    """

    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self.model = None
        self._feature_cols: list[str] | None = None
        self._scaler = None
        self.threshold: float = float("nan")
        self.history: dict | None = None

    # ── Training ────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, verbose: bool = False) -> "AutoEncoderDetector":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        set_seed(42)
        self._feature_cols = list(X.columns)
        self._scaler = StandardScaler().fit(X.values)
        Xs = self._scaler.transform(X.values).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_ae(Xs.shape[1], list(self.cfg.ae_hidden)).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.ae_lr)
        loss_fn = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs)),
            batch_size=self.cfg.ae_batch_size, shuffle=True,
        )
        losses: list[float] = []
        for ep in range(self.cfg.ae_epochs):
            self.model.train()
            ep_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(device)
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, xb)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item()) * xb.size(0)
            ep_loss /= len(loader.dataset)
            losses.append(ep_loss)
            if verbose:
                print(f"  AE epoch {ep+1:02d} loss={ep_loss:.5f}")

        # Calibrate threshold on training reconstruction errors
        with torch.no_grad():
            self.model.eval()
            recon = self.model(torch.from_numpy(Xs).to(device)).cpu().numpy()
        errs = np.mean((Xs - recon) ** 2, axis=1)
        self.threshold = float(np.quantile(errs, self.cfg.score_threshold_quantile))
        self.history = {"train_loss": losses, "recon_errors_summary": {
            "min": float(errs.min()), "max": float(errs.max()),
            "mean": float(errs.mean()), "p95": float(np.quantile(errs, 0.95)),
        }}
        return self

    # ── Inference ───────────────────────────────────────────────
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return the reconstruction-error score for each row (higher = anomaly)."""
        import torch
        if self.model is None or self._scaler is None:
            raise RuntimeError("Call fit() first.")
        Xs = self._scaler.transform(X[self._feature_cols].values).astype(np.float32)
        with torch.no_grad():
            self.model.eval()
            device = next(self.model.parameters()).device
            recon = self.model(torch.from_numpy(Xs).to(device)).cpu().numpy()
        return np.mean((Xs - recon) ** 2, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)

    # ── Persistence ─────────────────────────────────────────────
    def save(self) -> Path:
        out_dir = anomaly_model_dir()
        save_torch(self.model.state_dict(), out_dir / "autoencoder.pt")
        from src.ml.utils.io import save_joblib
        save_joblib(self._scaler, out_dir / "ae_scaler.joblib")
        write_json(out_dir / "ae_meta.json", {
            "feature_cols": self._feature_cols,
            "threshold": self.threshold,
            "cfg": vars(self.cfg),
            "history": self.history,
        })
        return out_dir / "autoencoder.pt"

    @classmethod
    def load(cls) -> "AutoEncoderDetector":
        out_dir = anomaly_model_dir()
        meta_path = out_dir / "ae_meta.json"
        weights_path = out_dir / "autoencoder.pt"
        if not (meta_path.exists() and weights_path.exists()):
            raise FileNotFoundError(weights_path)
        meta = read_json(meta_path)
        cfg = AnomalyConfig(**meta["cfg"])
        inst = cls(cfg)
        inst._feature_cols = meta["feature_cols"]
        inst.threshold = float(meta["threshold"])
        inst.history = meta.get("history")
        from src.ml.utils.io import load_joblib
        inst._scaler = load_joblib(out_dir / "ae_scaler.joblib")
        inst.model = _build_ae(len(inst._feature_cols), list(cfg.ae_hidden))
        inst.model.load_state_dict(load_torch(weights_path))
        inst.model.eval()
        return inst
