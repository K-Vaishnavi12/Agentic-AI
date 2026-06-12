"""Generate the master evaluation report (JSON + Markdown).

Reads from saved model cards (no retraining), optionally runs LLM eval if
``--with-llm`` is passed.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from src.ml.config import reports_dir, plots_dir
from src.ml.evaluation.eval_anomaly import collect_anomaly_metrics
from src.ml.evaluation.eval_forecasting import collect_forecast_metrics
from src.ml.evaluation.eval_legal import collect_legal_metrics
from src.ml.evaluation.eval_sentiment import collect_sentiment_metrics
from src.ml.utils.io import write_json


def _md_table_classification(report: dict) -> str:
    if not report:
        return "_(none)_"
    rows = ["| metric | value |", "|---|---|"]
    rows.append(f"| accuracy | {report.get('accuracy', 'n/a'):.3f} |")
    rows.append(f"| macro precision | {report.get('macro_precision', 'n/a'):.3f} |")
    rows.append(f"| macro recall | {report.get('macro_recall', 'n/a'):.3f} |")
    rows.append(f"| macro F1 | {report.get('macro_f1', 'n/a'):.3f} |")
    return "\n".join(rows)


def build_report(with_llm: bool = False) -> dict:
    forecast = collect_forecast_metrics()
    anomaly = collect_anomaly_metrics()
    sentiment = collect_sentiment_metrics()
    legal = collect_legal_metrics()
    bundle = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "forecasting": forecast,
        "anomaly": anomaly,
        "sentiment": sentiment,
        "legal": legal,
    }
    if with_llm:
        from src.ml.evaluation.eval_llm import evaluate_briefing
        bundle["llm_briefing"] = evaluate_briefing(verbose=True)

    # ── JSON ───────────────────────────────────────────────────
    json_path = reports_dir() / "eval_report.json"
    write_json(json_path, bundle)

    # ── Markdown ───────────────────────────────────────────────
    md_lines: list[str] = []
    md_lines.append("# Corporate Intelligence — ML Evaluation Report\n")
    md_lines.append(f"_Generated: {bundle['generated_at']}_\n")

    # Forecasting
    md_lines.append("## 1. Stock-price forecasting")
    fsum = forecast.get("summary") or {}
    if fsum:
        md_lines.append("")
        md_lines.append("| metric | LSTM (mean) | ARIMA (mean) |")
        md_lines.append("|---|---|---|")
        md_lines.append(f"| MAE | {fsum.get('lstm_mean_mae')} | {fsum.get('arima_mean_mae')} |")
        md_lines.append(f"| Directional accuracy | {fsum.get('lstm_mean_dir_acc')} | {fsum.get('arima_mean_dir_acc')} |")
        md_lines.append(f"| Symbols evaluated | {fsum.get('n_symbols_evaluated')} ||")
    md_lines.append("")
    md_lines.append("Per-symbol best model and metrics are in `eval_report.json` "
                    "and the per-symbol model cards under `models/forecast/<symbol>/model_card.json`.")
    md_lines.append("")

    # Anomaly
    md_lines.append("## 2. Anomaly detection")
    if anomaly.get("trained"):
        results = anomaly.get("results") or {}
        md_lines.append("")
        md_lines.append("| model | ROC-AUC | P@5% |")
        md_lines.append("|---|---|---|")
        for name, m in results.items():
            md_lines.append(f"| {name} | {m.get('roc_auc'):.3f} | {m.get('precision_at_5pct'):.3f} |")
        md_lines.append("")
        md_lines.append(f"**Best model:** `{anomaly.get('best')}`")
    else:
        md_lines.append("\n_Not trained yet. Run `python -m src.ml.anomaly.train_anomaly`._")
    md_lines.append("")

    # Sentiment
    md_lines.append("## 3. Financial sentiment classification")
    if sentiment.get("trained"):
        results = sentiment.get("results") or {}
        md_lines.append("")
        md_lines.append("**TF-IDF baseline:**\n")
        md_lines.append(_md_table_classification(results.get("tfidf_baseline") or {}))
        if results.get("distilbert"):
            md_lines.append("\n**DistilBERT (fine-tuned):**\n")
            md_lines.append(_md_table_classification(results["distilbert"]))
        md_lines.append(f"\n**Best:** `{sentiment.get('best')}`")
    else:
        md_lines.append("\n_Not trained. Run `python -m src.ml.sentiment.train_sentiment`._")
    md_lines.append("")

    # Legal
    md_lines.append("## 4. Legal text classification")
    if legal.get("trained"):
        results = legal.get("results") or {}
        md_lines.append("")
        md_lines.append("**TF-IDF baseline:**\n")
        md_lines.append(_md_table_classification(results.get("tfidf_baseline") or {}))
        if results.get("distilbert"):
            md_lines.append("\n**DistilBERT (fine-tuned):**\n")
            md_lines.append(_md_table_classification(results["distilbert"]))
        md_lines.append(f"\n**Best:** `{legal.get('best')}`")
    else:
        md_lines.append("\n_Not trained. Run `python -m src.ml.legal_classifier.train`._")
    md_lines.append("")

    # LLM
    if with_llm:
        llm = bundle.get("llm_briefing") or {}
        md_lines.append("## 5. LLM briefing evaluation")
        if "error" in llm:
            md_lines.append(f"\n_Skipped: {llm['error']}_")
        else:
            md_lines.append("")
            md_lines.append("| metric | value |")
            md_lines.append("|---|---|")
            md_lines.append(f"| latency (sec) | {llm.get('latency_seconds')} |")
            md_lines.append(f"| ROUGE-L vs sub-agent reference | {llm.get('rouge_l_vs_reference')} |")
            md_lines.append(f"| keyword coverage (macro) | {llm.get('coverage_macro')} |")
            md_lines.append(f"| briefing word count | {llm.get('briefing_words')} |")
        md_lines.append("")

    # Plots index
    md_lines.append("## Plots")
    pdir = plots_dir()
    if pdir.exists():
        for p in sorted(pdir.iterdir()):
            md_lines.append(f"- `reports/plots/{p.name}`")

    md_text = "\n".join(md_lines) + "\n"
    md_path = reports_dir() / "REPORT.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\n[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")
    return bundle


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="Also run an LLM briefing eval (needs Ollama running)")
    args = ap.parse_args()
    build_report(with_llm=args.with_llm)


if __name__ == "__main__":
    _cli()
