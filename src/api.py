"""FastAPI app — role-scoped access to agent outputs and executive briefing."""
import time
from pathlib import Path
from typing import Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from src.config import get_settings, Role, check_access, get_allowed_resources
from src.orchestrator import run_executive_briefing
from src.pipelines import get_context_from_pipelines

app = FastAPI(
    title="Corporate Intelligence API",
    description="Offline multi-agent platform — role-based access to agents and Chairman briefing.",
)

# ── Performance: cache agents & pipeline context ─────────────────────────────
_agents_cache: dict | None = None
_context_cache: dict | None = None
_context_cache_ts: float = 0.0
_CONTEXT_TTL = 60.0  # seconds


def _get_agents():
    global _agents_cache
    if _agents_cache is None:
        from src.agents import StockAgent, LegalAgent, StartupIntelligenceAgent, RiskAgent, ShareholderAgent
        _agents_cache = {
            "stock": StockAgent(),
            "legal": LegalAgent(),
            "startup": StartupIntelligenceAgent(),
            "risk": RiskAgent(),
            "shareholder": ShareholderAgent(),
        }
    return _agents_cache


def _get_context():
    global _context_cache, _context_cache_ts
    now = time.monotonic()
    if _context_cache is None or (now - _context_cache_ts) > _CONTEXT_TTL:
        _context_cache = get_context_from_pipelines()
        _context_cache_ts = now
    return _context_cache
# ─────────────────────────────────────────────────────────────────────────────

# Path to static dashboard (runs 100% offline)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = _STATIC_DIR / "index.html"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the offline dashboard UI."""
    if _DASHBOARD_HTML.exists():
        return FileResponse(_DASHBOARD_HTML)
    return HTMLResponse("<p>Dashboard not found. Run from project root.</p>", status_code=404)


class RoleQuery(BaseModel):
    role: Role


def get_role(role: str = Query(..., description="Stakeholder role")) -> Role:
    try:
        return Role(role.lower())
    except ValueError:
        raise HTTPException(422, detail=f"Invalid role. Allowed: {[r.value for r in Role]}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "offline"}


@app.get("/context-stats", summary="Pipeline counts for dashboard statistics")
def context_stats() -> dict[str, Any]:
    """Return counts from current pipeline data for each bot's statistics."""
    ctx = _get_context()
    stock = ctx.get("stock_data") or []
    legal = ctx.get("legal_data") or []
    startup = ctx.get("startup_data") or []
    shareholder = ctx.get("shareholder_data") or []
    rising = [s for s in stock if isinstance(s, dict) and s.get("change_pct", 0) > 0]
    falling = [s for s in stock if isinstance(s, dict) and s.get("change_pct", 0) < 0]
    return {
        "stock": {"total": len(stock), "rising": len(rising), "falling": len(falling)},
        "legal": {"total": len(legal), "compliance": len([x for x in legal if isinstance(x, dict) and x.get("type") == "compliance"])},
        "startup": {"total": len(startup), "sectors": len(set(x.get("sector") for x in startup if isinstance(x, dict) and x.get("sector")))},
        "risk": {"stock_count": len(stock), "legal_count": len(legal)},
        "shareholder": {"total": len(shareholder)},
    }


@app.get("/briefing", summary="Chairman only: full executive briefing")
def get_briefing(role: Role = Depends(get_role)) -> dict[str, Any]:
    if not check_access(role, "executive_briefing", get_settings().strict_rbac):
        raise HTTPException(403, detail="Access denied for this role to executive briefing.")
    context = _get_context()
    result = run_executive_briefing(context)
    return {
        "executive_briefing": result.get("executive_briefing"),
        "agent_outputs": result.get("agent_outputs", {}),
    }


# _get_agents() and _get_context() are defined above with caching


@app.get("/agent/{agent_id}", summary="Role-scoped agent output")
def get_agent_output(
    agent_id: str,
    role: Role = Depends(get_role),
    requirement: Optional[str] = Query(None, description="Optional user requirement/focus"),
) -> dict:
    if not check_access(role, f"{agent_id}_agent" if not agent_id.endswith("_agent") else agent_id, get_settings().strict_rbac):
        raise HTTPException(403, detail=f"Access denied for this role to agent {agent_id}.")
    context = _get_context()
    if requirement:
        context = {**context, "user_requirement": requirement}
    agents = _get_agents()
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(404, detail=f"Unknown agent: {agent_id}")
    return agent.run(context)


class AgentRunRequest(BaseModel):
    requirement: Optional[str] = None


@app.post("/agent/{agent_id}/run", summary="Run a single agent with optional requirement")
def run_agent(
    agent_id: str,
    body: AgentRunRequest,
    role: Role = Depends(get_role),
) -> dict:
    """Run one agent; pass requirement in JSON body to focus the output."""
    if not check_access(role, f"{agent_id}_agent" if not agent_id.endswith("_agent") else agent_id, get_settings().strict_rbac):
        raise HTTPException(403, detail=f"Access denied for this role to agent {agent_id}.")
    context = _get_context()
    if body.requirement:
        context = {**context, "user_requirement": body.requirement}
    agents = _get_agents()
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(404, detail=f"Unknown agent: {agent_id}")
    return agent.run(context)


@app.get("/allowed", summary="List resources allowed for a role")
def allowed_resources(role: Role = Depends(get_role)) -> dict:
    return {"role": role.value, "allowed_resources": list(get_allowed_resources(role))}


# ════════════════════════════════════════════════════════════════════
# ML endpoints — role-scoped, never crash if a model isn't trained yet
# ════════════════════════════════════════════════════════════════════

@app.get("/ml/status", summary="Which ML models are trained and ready")
def ml_status(role: Role = Depends(get_role)) -> dict:
    if not check_access(role, "stock_agent", get_settings().strict_rbac):
        # Most roles can see status; enforce a coarse permission check
        raise HTTPException(403, detail="Access denied to ML status")
    from src.ml.registry import model_status
    return model_status()


@app.get("/ml/forecast/{symbol}", summary="N-day price forecast for a symbol")
def ml_forecast(
    symbol: str,
    horizon: int = Query(5, ge=1, le=30, description="Forecast horizon (days)"),
    role: Role = Depends(get_role),
) -> dict:
    if not check_access(role, "stock_agent", get_settings().strict_rbac):
        raise HTTPException(403, detail="Access denied to forecasting")
    from src.ml.registry import get_forecaster
    f = get_forecaster(symbol.upper())
    if f is None:
        raise HTTPException(
            404,
            detail=f"No trained forecaster for {symbol}. "
                   f"Run: python -m src.ml.forecasting.train_forecast {symbol.upper()}",
        )
    try:
        result = f.predict(steps=horizon)
    except Exception as e:
        raise HTTPException(500, detail=f"Inference failed: {e}")
    return {
        "symbol": result.symbol,
        "model": result.model,
        "horizon_days": result.horizon,
        "last_close": result.last_close,
        "predicted_prices": result.predictions,
        "predicted_pct_changes": result.pct_changes,
        "direction": result.direction,
        "metrics": result.metrics,
        "trained_at": result.as_of,
    }


@app.get("/ml/anomaly", summary="Top anomalies across tracked symbols")
def ml_anomaly(
    lookback: int = Query(30, ge=5, le=120),
    top_k: int = Query(10, ge=1, le=50),
    role: Role = Depends(get_role),
) -> dict:
    if not check_access(role, "risk_agent", get_settings().strict_rbac):
        raise HTTPException(403, detail="Access denied to anomaly detection")
    from src.ml.registry import get_anomaly_detector
    det = get_anomaly_detector()
    if det is None:
        raise HTTPException(
            404,
            detail="No trained anomaly detector. "
                   "Run: python -m src.ml.anomaly.train_anomaly",
        )
    ctx = _get_context()
    syms = sorted({(s.get("symbol") or "").upper()
                   for s in (ctx.get("stock_data") or [])
                   if isinstance(s, dict) and s.get("symbol")})
    try:
        hits = det.detect(syms, lookback_days=lookback, top_k=top_k)
    except Exception as e:
        raise HTTPException(500, detail=f"Anomaly inference failed: {e}")
    return {
        "model": det.model_card.get("best") if det.model_card else None,
        "n_symbols": len(syms),
        "hits": [
            {
                "symbol": h.symbol, "date": h.date,
                "score": round(h.score, 3),
                "iso_flag": h.iso_flag, "ae_flag": h.ae_flag,
                "features": {k: round(v, 4) for k, v in h.features.items()},
            }
            for h in hits
        ],
    }


class SentimentRequest(BaseModel):
    texts: List[str]


@app.post("/ml/sentiment", summary="Score one or more texts (financial sentiment)")
def ml_sentiment(body: SentimentRequest, role: Role = Depends(get_role)) -> dict:
    if not check_access(role, "stock_agent", get_settings().strict_rbac):
        raise HTTPException(403, detail="Access denied to sentiment")
    if not body.texts:
        return {"predictions": []}
    from src.ml.registry import get_sentiment_classifier
    clf = get_sentiment_classifier()
    if clf is None:
        raise HTTPException(
            404,
            detail="No trained sentiment classifier. "
                   "Run: python -m src.ml.sentiment.train_sentiment",
        )
    try:
        preds = clf.predict(body.texts)
    except Exception as e:
        raise HTTPException(500, detail=f"Sentiment inference failed: {e}")
    return {
        "model": clf.model_name,
        "predictions": [
            {"text": p.text, "label": p.label,
             "confidence": round(p.confidence, 4),
             "probabilities": {k: round(v, 4) for k, v in p.probabilities.items()}}
            for p in preds
        ],
    }


class LegalClassifyRequest(BaseModel):
    texts: List[str]


@app.post("/ml/legal/classify", summary="Classify legal/regulatory texts")
def ml_legal_classify(body: LegalClassifyRequest, role: Role = Depends(get_role)) -> dict:
    if not check_access(role, "legal_agent", get_settings().strict_rbac):
        raise HTTPException(403, detail="Access denied to legal classifier")
    if not body.texts:
        return {"predictions": []}
    from src.ml.registry import get_legal_classifier
    clf = get_legal_classifier()
    if clf is None:
        raise HTTPException(
            404,
            detail="No trained legal classifier. "
                   "Run: python -m src.ml.legal_classifier.train",
        )
    try:
        preds = clf.predict(body.texts)
    except Exception as e:
        raise HTTPException(500, detail=f"Legal inference failed: {e}")
    return {
        "model": clf.model_name,
        "predictions": [
            {"text": p.text, "label": p.label,
             "confidence": round(p.confidence, 4),
             "probabilities": {k: round(v, 4) for k, v in p.probabilities.items()}}
            for p in preds
        ],
    }


@app.get("/ml/metrics", summary="Aggregate evaluation metrics for all ML modules")
def ml_metrics(role: Role = Depends(get_role)) -> dict:
    if not check_access(role, "executive_briefing", get_settings().strict_rbac):
        # Open to chairman / finance / investor for transparency on model quality
        if not any(check_access(role, r, get_settings().strict_rbac)
                   for r in ("stock_agent", "risk_agent", "legal_agent")):
            raise HTTPException(403, detail="Access denied to metrics")
    from src.ml.evaluation.run_all import build_report
    try:
        return build_report(with_llm=False)
    except Exception as e:
        raise HTTPException(500, detail=f"Metrics build failed: {e}")
