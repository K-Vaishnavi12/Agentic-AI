"""Risk Agent — cross-references financial volatility with legal exposure, flags danger zones.

ML enrichment:
  • Anomaly detector (IsolationForest + Autoencoder ensemble) flags symbols
    showing unusual return/volume signatures over the last lookback window.
  • Sentiment classifier scores any text snippets in legal/startup data so we
    can quantify *news* risk, not just price risk.

The legacy heuristic (change_pct < -3) is kept as a fallback when no model
is trained.
"""
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent
from src.ml.registry import get_anomaly_detector, get_sentiment_classifier


def _ml_anomaly_block(stock_data: list[dict]) -> dict | None:
    """Run the trained anomaly detector over symbols present in stock_data."""
    detector = get_anomaly_detector()
    if detector is None:
        return None
    symbols = sorted({(s.get("symbol") or "").upper()
                      for s in stock_data if isinstance(s, dict) and s.get("symbol")})
    if not symbols:
        return None
    try:
        hits = detector.detect(symbols, lookback_days=20, top_k=10)
    except Exception:
        return None
    return {
        "model_card": detector.model_card.get("best") if detector.model_card else None,
        "n_hits": len(hits),
        "hits": [
            {
                "symbol": h.symbol,
                "date": h.date,
                "score": round(h.score, 3),
                "iso_flag": h.iso_flag,
                "ae_flag": h.ae_flag,
                "ret_1d": round(h.features.get("ret_1d", 0.0), 4),
                "log_vol_z": round(h.features.get("log_vol_z", 0.0), 3),
            }
            for h in hits
        ],
    }


def _ml_news_sentiment(legal_data: list, startup_data: list | None = None) -> dict | None:
    """Score the sentiment of any text-bearing legal/startup events.

    Returns aggregate risk-relevant counters so the Risk Agent can flag a
    "negative news cluster" without LLM hallucination.
    """
    clf = get_sentiment_classifier()
    if clf is None:
        return None
    texts: list[tuple[str, str]] = []   # (source, text)
    for it in (legal_data or []):
        if not isinstance(it, dict):
            continue
        text = " ".join(filter(None, [it.get("title"), it.get("summary")])).strip()
        if text:
            texts.append(("legal", text))
    for it in (startup_data or []):
        if not isinstance(it, dict):
            continue
        text = it.get("description") or it.get("news") or ""
        if isinstance(text, str) and text.strip():
            texts.append(("startup", text.strip()))
    if not texts:
        return None
    try:
        preds = clf.predict([t for _, t in texts])
    except Exception:
        return None
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    items = []
    for (src, text), p in zip(texts, preds):
        counts[p.label] = counts.get(p.label, 0) + 1
        items.append({"source": src, "text": text[:160],
                      "label": p.label, "confidence": round(p.confidence, 3)})
    total = sum(counts.values()) or 1
    return {
        "model": clf.model_name,
        "n_scored": total,
        "counts": counts,
        "negative_share": round(counts["negative"] / total, 3),
        "items": items[:20],
    }


class RiskAgent(BaseAgent):
    agent_id = "risk"
    name = "Risk Agent"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        stock_data = context.get("stock_data", [])
        legal_data = context.get("legal_data", [])
        startup_data = context.get("startup_data", [])
        risk_input = {
            "stock": stock_data[:12],
            "legal": legal_data[:12],
        }
        if not stock_data and not legal_data:
            return {
                "agent": self.agent_id,
                "summary": "Insufficient data for risk analysis. Run stock and legal pipelines.",
                "danger_zones": [],
                "risk_score": "N/A",
                "ml_anomaly": None,
                "ml_news_sentiment": None,
            }

        # ── ML signals ─────────────────────────────────────────
        ml_anomaly = _ml_anomaly_block(stock_data)
        ml_sentiment = _ml_news_sentiment(legal_data, startup_data)

        req = context.get("user_requirement") or ""
        # Inject ML signals into the prompt so the LLM grounds its summary
        ml_brief = ""
        if ml_anomaly and ml_anomaly.get("hits"):
            top = ml_anomaly["hits"][:3]
            ml_brief += "ML anomaly detector flags: " + ", ".join(
                f"{h['symbol']} (score {h['score']})" for h in top
            ) + ". "
        if ml_sentiment:
            neg = ml_sentiment.get("negative_share", 0)
            ml_brief += (
                f"News sentiment: negative={ml_sentiment['counts'].get('negative', 0)} "
                f"({neg:.0%}), neutral={ml_sentiment['counts'].get('neutral', 0)}, "
                f"positive={ml_sentiment['counts'].get('positive', 0)}. "
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Risk Agent. Cross-reference stock volatility with legal data. "
             "If ML signals are provided, treat them as authoritative inputs and "
             "incorporate them into your risk summary. Be concise."),
            ("human",
             "Stock:\n{stock}\n\nLegal:\n{legal}\n\nML signals: {ml}\n\n"
             + ("User: {requirement}\n\n" if req else "")
             + "Brief risk summary and danger zones."),
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            "stock": str(risk_input["stock"]),
            "legal": str(risk_input["legal"]),
            "ml": ml_brief or "(no ML signals available; use heuristics)",
            "requirement": req,
        })

        # ── Combine heuristic + ML danger zones ───────────────
        danger_zones: list[dict] = []
        for s in stock_data:
            if isinstance(s, dict) and s.get("change_pct", 0) < -3:
                danger_zones.append({"source": "volatility", "item": s})
        if ml_anomaly:
            for h in ml_anomaly["hits"]:
                danger_zones.append({"source": "ml_anomaly",
                                      "item": {"symbol": h["symbol"],
                                               "score": h["score"],
                                               "date": h["date"]}})
        if ml_sentiment and ml_sentiment.get("negative_share", 0) >= 0.3:
            danger_zones.append({"source": "ml_news_sentiment",
                                  "item": {"negative_share": ml_sentiment["negative_share"],
                                           "n_scored": ml_sentiment["n_scored"]}})

        # Score: combines all three signals
        score_value = (
            len(danger_zones)
            + (ml_anomaly["n_hits"] if ml_anomaly else 0) * 0.5
            + (ml_sentiment["counts"]["negative"] if ml_sentiment else 0) * 0.5
        )
        if score_value >= 6:
            risk_label = "High"
        elif score_value >= 2:
            risk_label = "Medium"
        else:
            risk_label = "Low"

        result = {
            "agent": self.agent_id,
            "summary": response if isinstance(response, str) else getattr(response, "content", str(response)),
            "danger_zones": danger_zones[:10],
            "risk_score": risk_label,
            "ml_anomaly": ml_anomaly,
            "ml_news_sentiment": ml_sentiment,
        }
        self.remember(result["summary"], {"type": "risk_summary"})
        return result
