"""Sentiment training CLI: trains TF-IDF baseline always, then DistilBERT
unless ``ML_SKIP_TRANSFORMER`` is set. Picks the better one by macro-F1.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.ml.config import (
    SENTIMENT_LABELS, SentimentConfig, sentiment_model_dir, plots_dir,
)
from src.ml.data.sentiment_data import load_sentiment_dataset, train_test_split_text
from src.ml.sentiment.baseline_tfidf import TfidfSentimentBaseline
from src.ml.sentiment.finbert_classifier import FinSentimentClassifier
from src.ml.utils.io import write_json
from src.ml.utils.metrics import classification_report_dict
from src.ml.utils.plots import plot_confusion_matrix, plot_loss_curve
from src.ml.utils.seed import set_seed


def train(skip_transformer: bool | None = None, verbose: bool = True) -> dict:
    set_seed(42)
    settings = get_settings()
    if skip_transformer is None:
        skip_transformer = bool(settings.ml_skip_transformer)

    df = load_sentiment_dataset()
    train_df, test_df = train_test_split_text(df, test_size=0.2, seed=42)
    if verbose:
        print(f"[sentiment] train={len(train_df)} test={len(test_df)} "
              f"classes={dict(train_df['label'].value_counts())}")

    results: dict = {"n_train": int(len(train_df)), "n_test": int(len(test_df)),
                     "labels": SENTIMENT_LABELS}

    # ── TF-IDF baseline ────────────────────────────────────────
    base = TfidfSentimentBaseline().fit(
        train_df["text"].tolist(), train_df["label_id"].tolist(),
    )
    base.save()
    base_preds = base.predict(test_df["text"].tolist())
    base_report = classification_report_dict(
        test_df["label_id"].tolist(), base_preds, labels=SENTIMENT_LABELS,
    )
    results["tfidf_baseline"] = base_report
    if verbose:
        print(f"  TF-IDF: acc={base_report['accuracy']:.3f} "
              f"macro-F1={base_report['macro_f1']:.3f}")
    plot_confusion_matrix(
        base_report["confusion_matrix"], SENTIMENT_LABELS,
        out_path=plots_dir() / "sentiment_tfidf_cm.png",
        title="TF-IDF — confusion matrix",
    )

    # ── DistilBERT fine-tune ───────────────────────────────────
    bert_report = None
    if not skip_transformer:
        try:
            cfg = SentimentConfig(base_model=settings.ml_transformer_base)
            bert = FinSentimentClassifier(cfg)
            # 90/10 of train -> train/val for early stopping signal
            n_val = max(int(len(train_df) * 0.1), 16)
            tr = train_df.iloc[:-n_val]; va = train_df.iloc[-n_val:]
            bert.fit(
                tr["text"].tolist(), tr["label_id"].tolist(),
                val_texts=va["text"].tolist(), val_labels=va["label_id"].tolist(),
                verbose=verbose,
            )
            bert.save()
            bert_preds = bert.predict(test_df["text"].tolist())
            bert_report = classification_report_dict(
                test_df["label_id"].tolist(), bert_preds, labels=SENTIMENT_LABELS,
            )
            results["distilbert"] = bert_report
            if verbose:
                print(f"  DistilBERT: acc={bert_report['accuracy']:.3f} "
                      f"macro-F1={bert_report['macro_f1']:.3f}")
            plot_confusion_matrix(
                bert_report["confusion_matrix"], SENTIMENT_LABELS,
                out_path=plots_dir() / "sentiment_distilbert_cm.png",
                title="DistilBERT — confusion matrix",
            )
            if bert.history and bert.history.get("train_loss"):
                plot_loss_curve(
                    bert.history.get("train_loss") or [],
                    bert.history.get("val_loss") or [],
                    plots_dir() / "sentiment_distilbert_loss.png",
                    title="DistilBERT training loss",
                )
        except Exception as e:
            print(f"[sentiment] DistilBERT skipped: {e}")
            results["distilbert_error"] = str(e)
    else:
        print("[sentiment] DistilBERT skipped (ML_SKIP_TRANSFORMER set)")

    # ── Pick best ──────────────────────────────────────────────
    candidates = {"tfidf": base_report["macro_f1"]}
    if bert_report is not None:
        candidates["distilbert"] = bert_report["macro_f1"]
    best = max(candidates, key=candidates.get)
    results["best"] = best

    write_json(sentiment_model_dir() / "model_card.json", {
        "task": "financial_sentiment",
        "labels": SENTIMENT_LABELS,
        "best": best,
        "results": results,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return results


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformer", action="store_true",
                    help="Skip DistilBERT fine-tune (TF-IDF only)")
    args = ap.parse_args()
    train(skip_transformer=args.skip_transformer)


if __name__ == "__main__":
    _cli()
