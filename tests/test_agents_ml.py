"""End-to-end agent + ML pipeline tests (no LLM, no internet)."""
import os
os.environ["ML_SKIP_TRANSFORMER"] = "1"

from unittest.mock import MagicMock, patch

import pytest


def _fake_llm():
    """Stand-in LLM that returns a fixed string from invoke."""
    class _R:
        content = "Stub summary."
    fake = MagicMock()
    fake.invoke.return_value = _R()
    fake.stream.return_value = iter([_R()])
    return fake


def test_stock_agent_runs_without_forecaster_trained(monkeypatch):
    """Agent must still produce a result dict even if no ML model is trained."""
    from src.agents import stock_agent as sa
    monkeypatch.setattr(sa, "get_forecaster", lambda *_a, **_kw: None)

    # Patch the BaseAgent llm so we don't call Ollama
    agent = sa.StockAgent.__new__(sa.StockAgent)
    agent.llm = _fake_llm()
    agent.use_memory = False
    agent.agent_id = "stock"
    agent.name = "Stock Agent"
    # remember/recall are no-ops
    monkeypatch.setattr(agent, "remember", lambda *a, **kw: None)

    ctx = {
        "stock_data": [
            {"symbol": "SENSEX", "price": 100.0, "change_pct": 0.1, "volume": 0, "is_index": True},
            {"symbol": "RELIANCE", "price": 1000.0, "change_pct": -1.0, "volume": 1000, "is_index": False},
        ],
        "user_requirement": "reliance",
    }
    out = agent.run(ctx)
    assert out["agent"] == "stock"
    assert "forecast" in out  # field is present (None when not trained)
    assert out["requested_stock"]["symbol"] == "RELIANCE"


def test_legal_agent_runs_without_classifier_trained(monkeypatch):
    from src.agents import legal_agent as la
    monkeypatch.setattr(la, "get_legal_classifier", lambda *_a, **_kw: None)

    agent = la.LegalAgent.__new__(la.LegalAgent)
    agent.llm = _fake_llm()
    agent.use_memory = False
    agent.agent_id = "legal"
    agent.name = "Legal Agent"
    monkeypatch.setattr(agent, "remember", lambda *a, **kw: None)

    ctx = {
        "legal_data": [
            {"type": "compliance", "title": "Filing X due", "due_date": "2025-03-31"},
            {"type": "regulatory", "title": "SEBI norm Y", "summary": "new disclosure"},
        ]
    }
    out = agent.run(ctx)
    assert out["agent"] == "legal"
    assert "ml_classification" in out  # always present
    assert isinstance(out["compliance_items"], list)
    assert isinstance(out["regulatory_changes"], list)


def test_risk_agent_handles_missing_models(monkeypatch):
    from src.agents import risk_agent as ra
    monkeypatch.setattr(ra, "get_anomaly_detector", lambda *a, **kw: None)
    monkeypatch.setattr(ra, "get_sentiment_classifier", lambda *a, **kw: None)

    agent = ra.RiskAgent.__new__(ra.RiskAgent)
    agent.llm = _fake_llm()
    agent.use_memory = False
    agent.agent_id = "risk"
    agent.name = "Risk Agent"
    monkeypatch.setattr(agent, "remember", lambda *a, **kw: None)

    ctx = {
        "stock_data": [{"symbol": "RELIANCE", "price": 100, "change_pct": -4.0}],
        "legal_data": [],
        "startup_data": [],
    }
    out = agent.run(ctx)
    assert out["agent"] == "risk"
    assert "ml_anomaly" in out and out["ml_anomaly"] is None
    assert "ml_news_sentiment" in out
    # Heuristic: change_pct < -3 -> at least one danger zone
    assert len(out["danger_zones"]) >= 1
    assert out["risk_score"] in {"Low", "Medium", "High"}
