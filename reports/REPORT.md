# Corporate Intelligence — ML Evaluation Report

_Generated: 2026-06-11T15:52:48.471367+00:00_

## 1. Stock-price forecasting

| metric | LSTM (mean) | ARIMA (mean) |
|---|---|---|
| MAE | 64.0669769925963 | 63.6405078142699 |
| Directional accuracy | 0.4925830837255741 | 0.49292183711310805 |
| Symbols evaluated | 12 ||

Per-symbol best model and metrics are in `eval_report.json` and the per-symbol model cards under `models/forecast/<symbol>/model_card.json`.

## 2. Anomaly detection

| model | ROC-AUC | P@5% |
|---|---|---|
| isolation_forest | 1.000 | 0.966 |
| autoencoder | 1.000 | 1.000 |

**Best model:** `autoencoder`

## 3. Financial sentiment classification

**TF-IDF baseline:**

| metric | value |
|---|---|
| accuracy | 1.000 |
| macro precision | 1.000 |
| macro recall | 1.000 |
| macro F1 | 1.000 |

**DistilBERT (fine-tuned):**

| metric | value |
|---|---|
| accuracy | 0.931 |
| macro precision | 0.943 |
| macro recall | 0.931 |
| macro F1 | 0.930 |

**Best:** `tfidf`

## 4. Legal text classification

**TF-IDF baseline:**

| metric | value |
|---|---|
| accuracy | 1.000 |
| macro precision | 1.000 |
| macro recall | 1.000 |
| macro F1 | 1.000 |

**DistilBERT (fine-tuned):**

| metric | value |
|---|---|
| accuracy | 1.000 |
| macro precision | 1.000 |
| macro recall | 1.000 |
| macro F1 | 1.000 |

**Best:** `tfidf`

## Plots
- `reports/plots/anomaly_ae_loss.png`
- `reports/plots/anomaly_ae_scores.png`
- `reports/plots/anomaly_iso_scores.png`
- `reports/plots/forecast_BHARTIARTL_arima.png`
- `reports/plots/forecast_BHARTIARTL_lstm_loss.png`
- `reports/plots/forecast_HDFCBANK_arima.png`
- `reports/plots/forecast_HDFCBANK_lstm_loss.png`
- `reports/plots/forecast_HINDUNILVR_arima.png`
- `reports/plots/forecast_HINDUNILVR_lstm_loss.png`
- `reports/plots/forecast_ICICIBANK_arima.png`
- `reports/plots/forecast_ICICIBANK_lstm_loss.png`
- `reports/plots/forecast_INFY_arima.png`
- `reports/plots/forecast_INFY_lstm_loss.png`
- `reports/plots/forecast_ITC_arima.png`
- `reports/plots/forecast_ITC_lstm_loss.png`
- `reports/plots/forecast_KOTAKBANK_lstm.png`
- `reports/plots/forecast_KOTAKBANK_lstm_loss.png`
- `reports/plots/forecast_NIFTY50_arima.png`
- `reports/plots/forecast_NIFTY50_lstm_loss.png`
- `reports/plots/forecast_RELIANCE_lstm.png`
- `reports/plots/forecast_RELIANCE_lstm_loss.png`
- `reports/plots/forecast_SBIN_arima.png`
- `reports/plots/forecast_SBIN_lstm_loss.png`
- `reports/plots/forecast_SENSEX_arima.png`
- `reports/plots/forecast_SENSEX_lstm_loss.png`
- `reports/plots/forecast_TCS_arima.png`
- `reports/plots/forecast_TCS_lstm_loss.png`
- `reports/plots/legal_distilbert_cm.png`
- `reports/plots/legal_distilbert_loss.png`
- `reports/plots/legal_tfidf_cm.png`
- `reports/plots/sentiment_distilbert_cm.png`
- `reports/plots/sentiment_distilbert_loss.png`
- `reports/plots/sentiment_tfidf_cm.png`
