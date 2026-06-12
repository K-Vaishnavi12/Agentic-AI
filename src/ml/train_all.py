"""One-shot ML pipeline runner.

Usage:
    python -m src.ml.train_all                  # download data, train all, eval
    python -m src.ml.train_all --skip-data      # use cached data only
    python -m src.ml.train_all --skip-transformer  # baselines only

This is the main entry point for the report we ship to AMLSS reviewers.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import traceback


def _step(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true",
                    help="Don't refresh stock history; use cached files")
    ap.add_argument("--skip-transformer", action="store_true",
                    help="Skip DistilBERT fine-tuning (TF-IDF baselines only)")
    ap.add_argument("--with-llm-eval", action="store_true",
                    help="Also evaluate the orchestrator briefing (needs Ollama)")
    args = ap.parse_args()

    if args.skip_transformer:
        os.environ["ML_SKIP_TRANSFORMER"] = "1"

    t_start = time.time()
    failures: list[str] = []

    # ── 1. Data ────────────────────────────────────────────────
    if not args.skip_data:
        _step("1/5 Downloading / preparing stock history")
        try:
            from src.ml.data.stock_history import main as ensure_history_main
            ensure_history_main()
        except Exception as e:
            failures.append(f"stock_history: {e}")
            traceback.print_exc()
    else:
        _step("1/5 Skipping data refresh (--skip-data)")

    # ── 2. Forecasting ─────────────────────────────────────────
    _step("2/5 Training forecasting models (ARIMA + LSTM)")
    try:
        from src.ml.forecasting.train_forecast import train_all as train_forecast_all
        train_forecast_all()
    except Exception as e:
        failures.append(f"forecast: {e}")
        traceback.print_exc()

    # ── 3. Anomaly detection ───────────────────────────────────
    _step("3/5 Training anomaly detectors (IsolationForest + Autoencoder)")
    try:
        from src.ml.anomaly.train_anomaly import train_all as train_anomaly_all
        train_anomaly_all()
    except Exception as e:
        failures.append(f"anomaly: {e}")
        traceback.print_exc()

    # ── 4. Sentiment ───────────────────────────────────────────
    _step("4/5 Training sentiment classifier (TF-IDF + DistilBERT)")
    try:
        from src.ml.sentiment.train_sentiment import train as train_sentiment
        train_sentiment(skip_transformer=args.skip_transformer)
    except Exception as e:
        failures.append(f"sentiment: {e}")
        traceback.print_exc()

    # ── 4b. Legal classifier ───────────────────────────────────
    _step("4b/5 Training legal classifier (TF-IDF + DistilBERT)")
    try:
        from src.ml.legal_classifier.train import train as train_legal
        train_legal(skip_transformer=args.skip_transformer)
    except Exception as e:
        failures.append(f"legal: {e}")
        traceback.print_exc()

    # ── 5. Evaluation report ───────────────────────────────────
    _step("5/5 Building evaluation report")
    try:
        from src.ml.evaluation.run_all import build_report
        build_report(with_llm=args.with_llm_eval)
    except Exception as e:
        failures.append(f"eval: {e}")
        traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"Done in {elapsed:.1f}s.")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All steps OK. See `reports/REPORT.md`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
