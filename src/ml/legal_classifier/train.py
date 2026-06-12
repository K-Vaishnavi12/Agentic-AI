"""Train script for legal classifier — TF-IDF baseline + DistilBERT."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from src.config import get_settings
from src.ml.config import LEGAL_LABELS, LegalConfig, legal_model_dir, plots_dir
from src.ml.data.legal_data import load_legal_dataset, train_test_split_legal
from src.ml.legal_classifier.distilbert_clf import DistilBertLegalClassifier
from src.ml.legal_classifier.tfidf_logreg import TfidfLegalBaseline
from src.ml.utils.io import write_json
from src.ml.utils.metrics import classification_report_dict
from src.ml.utils.plots import plot_confusion_matrix, plot_loss_curve
from src.ml.utils.seed import set_seed


def train(skip_transformer: bool | None = None, verbose: bool = True) -> dict:
    set_seed(42)
    settings = get_settings()
    if skip_transformer is None:
        skip_transformer = bool(settings.ml_skip_transformer)

    df = load_legal_dataset()
    train_df, test_df = train_test_split_legal(df, test_size=0.2, seed=42)
    if verbose:
        print(f"[legal] train={len(train_df)} test={len(test_df)} "
              f"classes={dict(train_df['label'].value_counts())}")

    results: dict = {"n_train": int(len(train_df)), "n_test": int(len(test_df)),
                     "labels": LEGAL_LABELS}

    # ── Baseline ───────────────────────────────────────────────
    base = TfidfLegalBaseline().fit(
        train_df["text"].tolist(), train_df["label_id"].tolist(),
    )
    base.save()
    base_preds = base.predict(test_df["text"].tolist())
    base_report = classification_report_dict(
        test_df["label_id"].tolist(), base_preds, labels=LEGAL_LABELS,
    )
    results["tfidf_baseline"] = base_report
    if verbose:
        print(f"  TF-IDF acc={base_report['accuracy']:.3f} F1={base_report['macro_f1']:.3f}")
    plot_confusion_matrix(
        base_report["confusion_matrix"], LEGAL_LABELS,
        out_path=plots_dir() / "legal_tfidf_cm.png",
        title="Legal TF-IDF — confusion matrix",
    )

    # ── DistilBERT ─────────────────────────────────────────────
    bert_report = None
    if not skip_transformer:
        try:
            cfg = LegalConfig(base_model=settings.ml_transformer_base)
            bert = DistilBertLegalClassifier(cfg)
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
                test_df["label_id"].tolist(), bert_preds, labels=LEGAL_LABELS,
            )
            results["distilbert"] = bert_report
            if verbose:
                print(f"  DistilBERT acc={bert_report['accuracy']:.3f} F1={bert_report['macro_f1']:.3f}")
            plot_confusion_matrix(
                bert_report["confusion_matrix"], LEGAL_LABELS,
                out_path=plots_dir() / "legal_distilbert_cm.png",
                title="Legal DistilBERT — confusion matrix",
            )
            if bert.history and bert.history.get("train_loss"):
                plot_loss_curve(
                    bert.history.get("train_loss") or [],
                    bert.history.get("val_loss") or [],
                    plots_dir() / "legal_distilbert_loss.png",
                    title="Legal DistilBERT loss",
                )
        except Exception as e:
            print(f"[legal] DistilBERT skipped: {e}")
            results["distilbert_error"] = str(e)
    else:
        print("[legal] DistilBERT skipped (ML_SKIP_TRANSFORMER set)")

    candidates = {"tfidf": base_report["macro_f1"]}
    if bert_report is not None:
        candidates["distilbert"] = bert_report["macro_f1"]
    best = max(candidates, key=candidates.get)
    results["best"] = best
    write_json(legal_model_dir() / "model_card.json", {
        "task": "legal_text_classification",
        "labels": LEGAL_LABELS,
        "best": best,
        "results": results,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return results


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformer", action="store_true")
    args = ap.parse_args()
    train(skip_transformer=args.skip_transformer)


if __name__ == "__main__":
    _cli()
