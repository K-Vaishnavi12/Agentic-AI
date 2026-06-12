# Corporate Intelligence Platform

An offline-first, multi-agent AI platform for corporate stakeholders. Each
stakeholder (Chairman, investor, shareholder, legal, finance) has a dedicated
AI agent that fuses a local LLM with task-specific machine-learning models. A
LangGraph supervisor combines all agents into a single executive briefing.

The platform runs entirely on your machine. No cloud, no third-party data
exposure once trained.

### The problem it solves

Corporate decision-makers need a single pane of glass over market movements,
regulatory filings, competitor activity, and risk exposure. Existing tools
either ship data to a SaaS vendor (a non-starter for sensitive corporate
data) or rely on plain LLMs that hallucinate numbers. This project keeps all
data on-prem and grounds every claim in a trained model:

- **Stock predictions** come from ARIMA + LSTM models, not LLM guesses
- **Risk flags** come from an IsolationForest + Autoencoder ensemble
- **Sentiment and legal-text labels** come from fine-tuned DistilBERTs
- **The LLM only writes prose** around numbers the ML models produce

---

## TL;DR — five commands

```bash
pip install -r requirements.txt              # install
copy .env.example .env                       # configure (Windows; cp on *nix)
python -m src.ml.train_all                   # train all ML models (~3-6 min)
ollama pull phi3                             # local LLM for the briefing
uvicorn src.api:app --port 8000              # serve dashboard + API
```

Then open <http://127.0.0.1:8000>.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [Tech stack](#2-tech-stack)
3. [Project structure](#3-project-structure)
4. [How the pieces fit together](#4-how-the-pieces-fit-together)
5. [Setup](#5-setup)
6. [Running the platform](#6-running-the-platform)
7. [Training the ML models](#7-training-the-ml-models)
8. [API reference](#8-api-reference)
9. [Configuration](#9-configuration)
10. [Tests](#10-tests)
11. [Where to look first](#11-where-to-look-first)
12. [Troubleshooting](#troubleshooting)

---

## 1. What the project does

Five specialised agents, each focused on one stakeholder's view:

| Agent | Responsibility |
|---|---|
| **Stock Agent** | Reads market data, compares any stock to Sensex, attaches a 5-day price forecast |
| **Legal Agent** | Summarises MCA21 / SEBI / compliance items; auto-classifies them into compliance / regulatory / litigation / governance |
| **Startup Intelligence Agent** | New registrations, sectors, founders, funding rounds |
| **Risk Agent** | Cross-references stock volatility, legal exposure, news sentiment, and ML-flagged anomalies |
| **Shareholder Agent** | Dividends, equity movements, board decisions |

A **Chairman's Master Orchestrator** (LangGraph supervisor) runs all agents in
parallel and synthesises one unified executive briefing.

Underneath the agents, four ML modules do the actual quantitative work:

| ML module | Purpose | Models |
|---|---|---|
| Forecasting | Next-day & N-day stock prices | ARIMA + 2-layer LSTM (PyTorch) — one of each per symbol |
| Anomaly detection | Flag unusual return/volume signatures | IsolationForest + Dense Autoencoder ensemble |
| Financial sentiment | Score news/legal text as positive / neutral / negative | TF-IDF + Logistic Regression baseline + fine-tuned DistilBERT |
| Legal text classifier | 4-class compliance / regulatory / litigation / governance | TF-IDF + Logistic Regression baseline + fine-tuned DistilBERT |

A separate evaluation harness backtests every model and writes
`reports/REPORT.md` (Markdown) and `reports/eval_report.json` along with
plots under `reports/plots/`.

---

## 2. Tech stack

### Language & runtime
- Python 3.11
- Local Ollama (any chat model: `phi3`, `mistral`, `llama3`)

### Agent / orchestration
- **LangGraph** — supervisor graph and state management
- **LangChain** (`langchain`, `langchain-community`, `langchain-ollama`,
  `langchain-chroma`) — LLM and vector-store integrations
- **ChromaDB** — persistent vector memory per agent

### Web & API
- **FastAPI** + **Uvicorn** — HTTP server
- **Pydantic v2** + **pydantic-settings** — typed config and request models
- Plain HTML / CSS / JS dashboard (no JS framework, runs offline)

### Machine learning
- **NumPy**, **Pandas**, **SciPy**
- **scikit-learn** — IsolationForest, TF-IDF, LogisticRegression
- **statsmodels** — ARIMA
- **PyTorch** — LSTM forecaster, dense autoencoder, transformer fine-tuning
- **HuggingFace `transformers`** — DistilBERT, tokenizer
- **HuggingFace `datasets`** — Financial PhraseBank loader (with fallback)
- **`accelerate`**, **`evaluate`**, **`rouge-score`** — training helpers and
  metrics
- **Matplotlib**, **Seaborn** — plots written as PNGs

### Data
- **`yfinance`** — one-time download of 5 years of OHLCV history for the
  configured tickers (Sensex, Nifty 50, ten major NSE stocks)
- Local JSON files under `data/{stock,legal,startup}/` for offline operation

### Scheduling & utilities
- **APScheduler** — periodic data syncs
- **`joblib`** — sklearn artifact persistence
- **`pytest`** — test runner

---

## 3. Project structure

```
.
├── data/                       # Raw & processed data (offline)
│   ├── chroma/                 # ChromaDB persistence
│   ├── stock/stock.json
│   ├── legal/legal.json
│   ├── startup/startups.json
│   └── ml/                     # Training datasets cached here
│       ├── stock_history/      # Yahoo Finance CSV per symbol
│       ├── sentiment/financial_phrasebank.csv
│       └── legal/legal_corpus.csv
├── models/                     # Trained model artifacts
│   ├── forecast/<symbol>/      # arima.joblib + lstm.pt + model_card.json
│   ├── anomaly/                # isolation_forest.joblib + autoencoder.pt
│   ├── sentiment/              # tfidf_baseline.joblib + distilbert/
│   └── legal/                  # tfidf_baseline.joblib + distilbert/
├── reports/                    # Auto-generated by the eval harness
│   ├── REPORT.md
│   ├── eval_report.json
│   └── plots/                  # All PNGs (loss curves, confusion matrices…)
├── src/
│   ├── config/                 # Settings (env-driven) + RBAC roles
│   ├── llm/                    # Ollama wrapper
│   ├── memory/                 # ChromaDB persistent memory
│   ├── agents/                 # Stock / Legal / Startup / Risk / Shareholder
│   ├── orchestrator/           # LangGraph supervisor → Chairman briefing
│   ├── pipelines/              # Data sync (stock, legal, startup) + Yahoo
│   ├── api.py                  # FastAPI app with all endpoints
│   ├── main.py                 # CLI entry point
│   ├── static/                 # Offline dashboard (HTML/CSS/JS)
│   └── ml/                     # ── ML stack ──
│       ├── config.py           # Hyperparameters, label spaces, paths
│       ├── registry.py         # Single entry point for loading any model
│       ├── train_all.py        # One-shot trainer: data → models → report
│       ├── data/               # Loaders (stock history, sentiment, legal)
│       ├── forecasting/        # ARIMA + LSTM
│       ├── anomaly/            # IsolationForest + Autoencoder
│       ├── sentiment/          # TF-IDF + DistilBERT
│       ├── legal_classifier/   # TF-IDF + DistilBERT
│       ├── evaluation/         # Metric collectors + report builder
│       └── utils/              # Seed, metrics, plots, IO
├── tests/                      # 23 unit + integration tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. How the pieces fit together

```
                      ┌─────────────────────────────┐
                      │  Pipelines (stock / legal / │
                      │  startup data sync)         │
                      └──────────────┬──────────────┘
                                     │ context dict
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                         ▼
     ┌────────────┐          ┌────────────┐            ┌────────────┐
     │ Stock      │          │ Legal      │            │ Risk       │
     │ Agent      │          │ Agent      │            │ Agent      │
     │            │          │            │            │            │
     │ + Forecast │          │ + Legal    │            │ + Anomaly  │
     │   model    │          │   classifier│           │   ensemble │
     │            │          │            │            │ + Sentiment│
     └─────┬──────┘          └─────┬──────┘            └─────┬──────┘
           │                       │                         │
           └───────────┬───────────┴────────────┬────────────┘
                       │                        │
                       ▼                        ▼
                 ┌────────────────────────────────────┐
                 │  LangGraph Supervisor              │
                 │  (Chairman's Master Orchestrator)  │
                 └──────────────────┬─────────────────┘
                                    ▼
                          ┌──────────────────────┐
                          │  Executive Briefing  │
                          │  (single LLM output) │
                          └──────────────────────┘
```

Every agent is a `BaseAgent` with an LLM and optional ChromaDB memory. ML
models are loaded lazily through `src.ml.registry` so the agents stay
ignorant of where artifacts live or which framework produced them. If a model
hasn't been trained yet, the agent gracefully falls back to its rule-based
behaviour.

---

## 5. Setup

### 5.1 Clone & Python environment

```bash
python -m venv .venv

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)  — if blocked, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ is required; this project was developed and tested on 3.11.

### 5.2 Configuration

```bash
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux
```

Defaults are sensible. Key options live in `src/config/settings.py` and
include `OLLAMA_MODEL`, `MODELS_DIR`, `ML_LOOKBACK`, `ML_HORIZON`, and
`ML_TRANSFORMER_BASE`. See [Configuration](#9-configuration) below.

### 5.3 Local LLM (optional, only for the orchestrator briefing)

The ML modules and dashboard work without an LLM. The Chairman briefing and
agent prose summaries do need Ollama:

```bash
ollama pull phi3                  # ~2.3 GB, recommended on small machines
ollama pull nomic-embed-text      # used for ChromaDB embeddings
```

Set `OLLAMA_MODEL=phi3` (or `mistral`, `llama3`, etc.) in `.env`.

### 5.4 Recommended first run

For a brand-new clone, this sequence gets you to a working dashboard in
about ten minutes:

```bash
# 1. Environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# 2. Train the ML stack (downloads data, trains, writes a report)
python -m src.ml.train_all

# 3. Start Ollama in a separate terminal, then pull a model
ollama serve
ollama pull phi3

# 4. Run the dashboard
uvicorn src.api:app --port 8000
```

After step 2 you'll have:
- `models/` populated with 12 forecasters + 4 classifiers + an anomaly ensemble
- `reports/REPORT.md` with metrics and `reports/plots/` with PNGs
- `data/ml/stock_history/` with five years of OHLCV per symbol

---

## 6. Running the platform

### 6.1 CLI: print one executive briefing

```bash
python -m src.main
```

This loads pipeline data, runs every agent in parallel, and streams the
final Chairman briefing to stdout.

### 6.2 Web dashboard + REST API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The dashboard is plain HTML/CSS/JS — no build
step, no internet needed at runtime. From there you can run a briefing,
inspect each agent, and (with models trained) view forecasts and
anomalies.

### 6.3 Refresh stock data (one-time, needs internet)

```bash
python -m src.pipelines.stock_api
```

Downloads the latest Sensex / Nifty / stock prices to
`data/stock/stock.json`. Needed only when you want fresh prices; the
ML pipeline caches its own 5-year history separately.

---

## 7. Training the ML models

### 7.1 One command to train everything

```bash
python -m src.ml.train_all
```

This sequentially:
1. Downloads 5 years of daily OHLCV from Yahoo Finance for 12 symbols
   (or synthesises realistic data if offline).
2. Trains 12 ARIMA + 12 LSTM forecasters with walk-forward backtests.
3. Trains the IsolationForest + Autoencoder anomaly ensemble on
   pooled features (~14k rows).
4. Trains the TF-IDF baseline and fine-tunes DistilBERT for sentiment.
5. Trains the TF-IDF baseline and fine-tunes DistilBERT for legal text.
6. Builds `reports/REPORT.md` and `reports/eval_report.json`.

Total runtime: about 3–6 minutes on CPU. Disk usage: ~600 MB (mostly
DistilBERT weights).

### 7.2 Faster paths

```bash
# Skip data download (use cached files)
python -m src.ml.train_all --skip-data

# Skip transformer fine-tuning (TF-IDF baselines only)
python -m src.ml.train_all --skip-transformer

# Also score the LLM briefing (needs Ollama running)
python -m src.ml.train_all --with-llm-eval
```

### 7.3 Train one module at a time

```bash
python -m src.ml.data.stock_history             # download / cache history
python -m src.ml.forecasting.train_forecast     # all symbols
python -m src.ml.forecasting.train_forecast SENSEX TCS   # specific
python -m src.ml.anomaly.train_anomaly
python -m src.ml.sentiment.train_sentiment
python -m src.ml.legal_classifier.train
python -m src.ml.evaluation.run_all             # rebuild reports only
```

### 7.4 What gets saved

- `models/forecast/<symbol>/` — `arima.joblib`, `lstm.pt`, `lstm_meta.json`,
  `model_card.json` (config + metrics + timestamp)
- `models/anomaly/` — IsolationForest + autoencoder + threshold + card
- `models/sentiment/` — `tfidf_baseline.joblib` + `distilbert/` + card
- `models/legal/` — same shape as sentiment
- `reports/plots/` — loss curves, confusion matrices, forecast plots, anomaly
  score distributions

---

## 8. API reference

All endpoints are role-scoped. Pass `?role=<chairman|investor|shareholder|legal|finance>`.

### Existing endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/` | Offline dashboard UI |
| GET | `/context-stats` | Counts from current pipeline data |
| GET | `/briefing?role=chairman` | Chairman: full executive briefing |
| GET | `/agent/{agent_id}?role=...` | Single agent's role-scoped output |
| POST | `/agent/{agent_id}/run?role=...` | Same with optional `requirement` body |
| GET | `/allowed?role=...` | List resources allowed for a role |

### ML endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/ml/status` | Which models are trained and ready |
| GET | `/ml/forecast/{symbol}?horizon=5` | N-day price forecast + backtest metrics |
| GET | `/ml/anomaly?lookback=30&top_k=10` | Top anomalous symbol/days from the ensemble |
| POST | `/ml/sentiment` `{"texts": [...]}` | Financial sentiment with confidences |
| POST | `/ml/legal/classify` `{"texts": [...]}` | 4-class legal classification |
| GET | `/ml/metrics` | Aggregated metrics across all ML modules |

Example:

```bash
curl "http://127.0.0.1:8000/ml/forecast/RELIANCE?role=chairman&horizon=5"
```

```json
{
  "symbol": "RELIANCE",
  "model": "lstm",
  "horizon_days": 5,
  "last_close": 1263.0,
  "predicted_prices": [1263.45, 1263.88, 1264.29, 1264.7, 1265.11],
  "predicted_pct_changes": [0.035, 0.034, 0.033, 0.032, 0.032],
  "direction": "up",
  "metrics": {"mae": 13.18, "rmse": 17.99, "mape_pct": 0.93, "directional_accuracy": 0.45},
  "trained_at": "2026-06-11T..."
}
```

---

## 9. Configuration

Set in `.env` (read by `src/config/settings.py`).

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `phi3` | Chat model (try `mistral`, `llama3`) |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB on-disk store |
| `DATA_DIR` | `./data` | Root data dir |
| `STOCK_DATA_PATH` | `./data/stock` | Pipeline input |
| `LEGAL_DATA_PATH` | `./data/legal` | Pipeline input |
| `STARTUP_DATA_PATH` | `./data/startup` | Pipeline input |
| `STRICT_RBAC` | `true` | Enforce role-based access |
| `MODELS_DIR` | `./models` | Trained ML artifacts |
| `REPORTS_DIR` | `./reports` | Auto-generated reports |
| `ML_DATA_DIR` | `./data/ml` | Cached ML datasets |
| `ML_SEED` | `42` | Single seed for all ML training |
| `ML_LOOKBACK` | `60` | LSTM input window (days) |
| `ML_HORIZON` | `5` | Default forecast horizon (days) |
| `ML_TRANSFORMER_BASE` | `distilbert-base-uncased` | HF model for fine-tuning |
| `ML_SKIP_TRANSFORMER` | `0` | Set to `1` to use TF-IDF baseline only |

### Roles & access

| Role | Allowed resources |
|---|---|
| `chairman` | Everything: full briefing + every ML endpoint |
| `investor` | Stock + Risk agents, forecasting, anomalies |
| `shareholder` | Shareholder agent (dividends, equity, board) |
| `legal` | Legal agent + legal classifier + regulatory data |
| `finance` | Risk + Stock + ML metrics |

Edit `src/config/rbac.py` to adjust.

---

## 10. Tests

```bash
pytest tests/ -q
```

Covers:
- ML metric implementations (MAE, RMSE, MAPE, directional accuracy, F1)
- Reproducibility (seeds yield identical numpy draws)
- Forecasting: window construction, LSTM fit and forecast smoke,
  walk-forward shape
- Anomaly: feature engineering, IsolationForest, autoencoder
- Classifiers: dataset balance, TF-IDF accuracy ≫ random
- Agent integration: each agent's `run()` works whether or not the
  corresponding ML model is trained

All 23 tests pass on a clean checkout (with `ML_SKIP_TRANSFORMER=1` so they
finish in ~13 seconds).

---

## 11. Where to look first

If you're reading the code for the first time, the most informative paths
are:

| Goal | File |
|---|---|
| How an agent is structured | `src/agents/base.py` |
| How agents are orchestrated | `src/orchestrator/supervisor.py` |
| How ML models plug into agents | `src/ml/registry.py` and `src/agents/risk_agent.py` |
| How forecasting works end-to-end | `src/ml/forecasting/train_forecast.py` |
| How a transformer is fine-tuned | `src/ml/sentiment/finbert_classifier.py` |
| How metrics are computed | `src/ml/utils/metrics.py` |
| How the report is generated | `src/ml/evaluation/run_all.py` |
| API surface | `src/api.py` |
| Settings | `src/config/settings.py` |
| Tests | `tests/` |

---

## License & data

All processing and storage stay on your server. No external APIs at runtime
once trained. Yahoo Finance data and the Financial PhraseBank are fetched
once during the initial training run and cached locally; both are
freely available for research use under their respective terms.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'matplotlib'`** (or similar) after
running training: install all dependencies with
`pip install -r requirements.txt`. Behind a corporate proxy you may need
`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`.

**`Connection refused` when running the briefing:** Ollama isn't running.
Start it with `ollama serve` in a separate terminal and confirm the model is
pulled (`ollama list`).

**Training fails on `yfinance`:** Yahoo Finance can be flaky. The data
loader transparently falls back to a geometric-Brownian-motion synthetic
series so training still succeeds offline. Re-run `python -m
src.ml.data.stock_history` later to refresh real data.

**DistilBERT step is slow / running out of memory:** set
`ML_SKIP_TRANSFORMER=1` in `.env` (or pass `--skip-transformer` to
`train_all`). The platform falls back to the TF-IDF + Logistic Regression
baselines, which are nearly as accurate on this corpus.

**Tests timeout:** the transformer-heavy tests are skipped by default via
`ML_SKIP_TRANSFORMER=1` in `tests/conftest.py`. The full suite runs in about
13 seconds.

---

## Credits

- **LangGraph / LangChain** — multi-agent orchestration
- **Ollama** — local LLM runtime
- **HuggingFace `transformers` / `datasets`** — DistilBERT and corpora
- **scikit-learn**, **statsmodels**, **PyTorch** — ML backbone
- **Yahoo Finance** (`yfinance`) — historical market data
- **Financial PhraseBank** (Malo et al., 2014) — sentiment labels
