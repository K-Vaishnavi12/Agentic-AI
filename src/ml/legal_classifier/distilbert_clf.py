"""DistilBERT 4-class classifier for legal text."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np

from src.ml.config import (
    LEGAL_LABELS, LEGAL_LABEL2ID, LEGAL_ID2LABEL,
    LegalConfig, legal_model_dir,
)
from src.ml.utils.io import write_json, read_json
from src.ml.utils.seed import set_seed


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _build_model(name: str):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(LEGAL_LABELS),
        id2label=LEGAL_ID2LABEL, label2id=LEGAL_LABEL2ID,
    )


class DistilBertLegalClassifier:
    def __init__(self, cfg: LegalConfig | None = None):
        self.cfg = cfg or LegalConfig()
        self.model = None
        self.tokenizer = None
        self.history: dict | None = None

    def fit(self, train_texts: Sequence[str], train_labels: Sequence[int],
            val_texts: Sequence[str] | None = None,
            val_labels: Sequence[int] | None = None,
            verbose: bool = True) -> "DistilBertLegalClassifier":
        import torch
        from torch.utils.data import DataLoader

        set_seed(42)
        self.tokenizer = _load_tokenizer(self.cfg.base_model)
        self.model = _build_model(self.cfg.base_model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        def encode(texts):
            return self.tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=self.cfg.max_len, return_tensors="pt",
            )

        class _DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc, self.labels = enc, labels
            def __len__(self): return len(self.labels)
            def __getitem__(self, i):
                return ({k: v[i] for k, v in self.enc.items()}, self.labels[i])

        train_loader = DataLoader(
            _DS(encode(train_texts), torch.tensor(np.asarray(train_labels), dtype=torch.long)),
            batch_size=self.cfg.batch_size, shuffle=True,
        )
        val_loader = None
        if val_texts is not None and val_labels is not None and len(val_texts):
            val_loader = DataLoader(
                _DS(encode(val_texts), torch.tensor(np.asarray(val_labels), dtype=torch.long)),
                batch_size=self.cfg.batch_size, shuffle=False,
            )

        from torch.optim import AdamW
        opt = AdamW(self.model.parameters(), lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay)

        train_losses, val_losses, val_accs = [], [], []
        for ep in range(self.cfg.epochs):
            self.model.train()
            ep_loss = 0.0
            for bx, by in train_loader:
                for k in bx: bx[k] = bx[k].to(device)
                by = by.to(device)
                opt.zero_grad()
                out = self.model(**bx, labels=by)
                out.loss.backward()
                opt.step()
                ep_loss += float(out.loss.item()) * by.size(0)
            ep_loss /= len(train_loader.dataset)
            train_losses.append(ep_loss)

            v_loss = float("nan"); v_acc = float("nan")
            if val_loader is not None:
                self.model.eval()
                tot, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        for k in bx: bx[k] = bx[k].to(device)
                        by = by.to(device)
                        out = self.model(**bx, labels=by)
                        tot += float(out.loss.item()) * by.size(0)
                        preds = out.logits.argmax(-1)
                        correct += int((preds == by).sum().item()); total += by.size(0)
                v_loss = tot / max(1, total); v_acc = correct / max(1, total)
                val_losses.append(v_loss); val_accs.append(v_acc)
            if verbose:
                print(f"  legal-bert epoch {ep+1}/{self.cfg.epochs} "
                      f"train={ep_loss:.4f} val={v_loss:.4f} acc={v_acc:.3f}")
        self.history = {"train_loss": train_losses,
                        "val_loss": val_losses, "val_acc": val_accs}
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        self.model.eval()
        with torch.no_grad():
            return self.model(**enc).logits.argmax(-1).cpu().numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()

    def save(self) -> Path:
        out = legal_model_dir() / "distilbert"
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        write_json(out / "history.json", self.history or {})
        return out

    @classmethod
    def load(cls) -> "DistilBertLegalClassifier":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        out = legal_model_dir() / "distilbert"
        if not out.exists():
            raise FileNotFoundError(out)
        inst = cls()
        inst.tokenizer = AutoTokenizer.from_pretrained(out)
        inst.model = AutoModelForSequenceClassification.from_pretrained(out)
        inst.model.eval()
        inst.history = read_json(out / "history.json", default={})
        return inst
