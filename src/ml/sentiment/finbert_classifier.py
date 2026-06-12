"""DistilBERT financial sentiment classifier (fine-tuned on Financial PhraseBank).

We avoid the HuggingFace ``Trainer`` to keep dependencies thin and the loop
transparent. AMP off by default for CPU compatibility; flip on if CUDA.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np

from src.ml.config import (
    SENTIMENT_LABELS, SENTIMENT_LABEL2ID, SENTIMENT_ID2LABEL,
    SentimentConfig, sentiment_model_dir,
)
from src.ml.utils.io import write_json, read_json
from src.ml.utils.seed import set_seed


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _build_model(name: str, num_labels: int = 3):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=num_labels,
        id2label=SENTIMENT_ID2LABEL, label2id=SENTIMENT_LABEL2ID,
    )


class FinSentimentClassifier:
    def __init__(self, cfg: SentimentConfig | None = None):
        self.cfg = cfg or SentimentConfig()
        self.model = None
        self.tokenizer = None
        self.history: dict | None = None

    def fit(self, train_texts: Sequence[str], train_labels: Sequence[int],
            val_texts: Sequence[str] | None = None,
            val_labels: Sequence[int] | None = None,
            verbose: bool = True) -> "FinSentimentClassifier":
        import torch
        from torch.utils.data import DataLoader

        set_seed(42)
        self.tokenizer = _load_tokenizer(self.cfg.base_model)
        self.model = _build_model(self.cfg.base_model, num_labels=len(SENTIMENT_LABELS))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        def encode(texts: Sequence[str]):
            return self.tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=self.cfg.max_len, return_tensors="pt",
            )

        train_enc = encode(train_texts)
        train_labels_t = torch.tensor(np.asarray(train_labels), dtype=torch.long)

        class _DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc, self.labels = enc, labels
            def __len__(self): return len(self.labels)
            def __getitem__(self, i):
                return ({k: v[i] for k, v in self.enc.items()}, self.labels[i])

        train_loader = DataLoader(
            _DS(train_enc, train_labels_t),
            batch_size=self.cfg.batch_size, shuffle=True,
        )
        val_loader = None
        val_labels_t = None
        if val_texts is not None and val_labels is not None and len(val_texts):
            val_enc = encode(val_texts)
            val_labels_t = torch.tensor(np.asarray(val_labels), dtype=torch.long)
            val_loader = DataLoader(_DS(val_enc, val_labels_t),
                                    batch_size=self.cfg.batch_size, shuffle=False)

        # Optimizer + scheduler
        from torch.optim import AdamW
        opt = AdamW(self.model.parameters(), lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay)
        total_steps = max(1, len(train_loader) * self.cfg.epochs)
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        def lr_at(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        train_losses, val_losses, val_accs = [], [], []
        global_step = 0
        for ep in range(self.cfg.epochs):
            self.model.train()
            ep_loss = 0.0
            for batch_x, batch_y in train_loader:
                for k in batch_x:
                    batch_x[k] = batch_x[k].to(device)
                batch_y = batch_y.to(device)
                # Manual LR scaling per step
                for g in opt.param_groups:
                    g["lr"] = self.cfg.lr * lr_at(global_step)
                opt.zero_grad()
                out = self.model(**batch_x, labels=batch_y)
                out.loss.backward()
                opt.step()
                ep_loss += float(out.loss.item()) * batch_y.size(0)
                global_step += 1
            ep_loss /= len(train_loader.dataset)
            train_losses.append(ep_loss)

            v_loss = float("nan"); v_acc = float("nan")
            if val_loader is not None:
                self.model.eval()
                tot_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        for k in bx: bx[k] = bx[k].to(device)
                        by = by.to(device)
                        out = self.model(**bx, labels=by)
                        tot_loss += float(out.loss.item()) * by.size(0)
                        preds = out.logits.argmax(dim=-1)
                        correct += int((preds == by).sum().item())
                        total += int(by.size(0))
                v_loss = tot_loss / max(1, total)
                v_acc = correct / max(1, total)
                val_losses.append(v_loss); val_accs.append(v_acc)
            if verbose:
                print(f"  epoch {ep+1}/{self.cfg.epochs} train_loss={ep_loss:.4f} "
                      f"val_loss={v_loss:.4f} val_acc={v_acc:.3f}")

        self.history = {"train_loss": train_losses,
                        "val_loss": val_losses, "val_acc": val_accs}
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call fit() first.")
        device = next(self.model.parameters()).device
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**enc).logits
        return logits.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def save(self) -> Path:
        out = sentiment_model_dir() / "distilbert"
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        write_json(out / "history.json", self.history or {})
        return out

    @classmethod
    def load(cls) -> "FinSentimentClassifier":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        out = sentiment_model_dir() / "distilbert"
        if not out.exists():
            raise FileNotFoundError(out)
        inst = cls()
        inst.tokenizer = AutoTokenizer.from_pretrained(out)
        inst.model = AutoModelForSequenceClassification.from_pretrained(out)
        inst.model.eval()
        inst.history = read_json(out / "history.json", default={})
        return inst
