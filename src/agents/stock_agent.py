"""Stock Agent — monitors market data, compares to Sensex, returns particular stock value.

ML enrichment: when an LSTM/ARIMA forecast model is trained for the requested
symbol (or any symbol in `data/stock/stock.json`), the agent attaches a
``forecast`` block with N-day price predictions, directional signal, and the
training-time backtest metrics.
"""
from typing import Any, Iterator
import re
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent
from src.ml.registry import get_forecaster


# Map user query keywords to possible symbols (case-insensitive)
SYMBOL_ALIASES = {
    "sensex": "SENSEX",
    "nifty": "NIFTY50",
    "gold": "GOLD",
    "reliance": "RELIANCE",
    "tcs": "TCS",
    "infy": "INFY",
    "infosys": "INFY",
    "acme": "ACME",
    "globex": "GLOBEX",
    "initech": "INITECH",
    "cyberdyn": "CYBERDYN",
    "omnicorp": "OMNICORP",
}


def _find_requested_stock(stock_data: list, requirement: str) -> dict | None:
    """Find the stock the user asked for (by symbol or keyword). Any symbol in data can be asked."""
    if not requirement or not stock_data:
        return None
    req_lower = requirement.lower().strip()
    # Try direct symbol match first (any stock in data: e.g. "gold", "reliance", "TCS", "CYBERDYN")
    for s in stock_data:
        if not isinstance(s, dict):
            continue
        sym = (s.get("symbol") or "").upper()
        if not sym:
            continue
        sym_lower = sym.lower()
        if sym_lower in req_lower or req_lower in sym_lower:
            return s
    # Try keyword/alias match
    for word in re.findall(r"[a-z0-9]+", req_lower):
        if len(word) < 2:
            continue
        sym = SYMBOL_ALIASES.get(word)
        if sym:
            for s in stock_data:
                if isinstance(s, dict) and (s.get("symbol") or "").upper() == sym:
                    return s
    # Try partial symbol match (e.g. "rel" -> RELIANCE, "inf" -> INFY)
    for s in stock_data:
        if not isinstance(s, dict):
            continue
        sym = (s.get("symbol") or "").lower()
        if sym and (sym in req_lower or (req_lower in sym and len(req_lower) >= 2)):
            return s
    return None


def _get_sensex(stock_data: list) -> dict | None:
    """Get Sensex index from data."""
    for s in stock_data:
        if isinstance(s, dict) and (s.get("symbol") or "").upper() == "SENSEX":
            return s
    return None


def _ml_forecast(symbol: str, steps: int = 5) -> dict | None:
    """Return a forecast dict for the given symbol, or None if no model."""
    if not symbol:
        return None
    forecaster = get_forecaster(symbol.upper())
    if forecaster is None:
        return None
    try:
        result = forecaster.predict(steps=steps)
    except Exception:
        return None
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


class StockAgent(BaseAgent):
    agent_id = "stock"
    name = "Stock Agent"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        stock_data = context.get("stock_data", [])
        if not stock_data:
            return {
                "agent": self.agent_id,
                "summary": "No stock data available. Run the stock pipeline to sync market data.",
                "alerts": [],
                "rising": [],
                "falling": [],
                "chart_data": [],
                "all_stocks": [],
                "requested_stock": None,
                "sensex_comparison": None,
            }

        req = (context.get("user_requirement") or "").strip()
        sensex = _get_sensex(stock_data)
        requested = _find_requested_stock(stock_data, req) if req else None

        # Build comparison text for particular stock vs Sensex
        requested_stock = None
        sensex_comparison = None
        if requested and sensex:
            sp = float(requested.get("price", 0))
            sc = float(requested.get("change_pct", 0))
            sensex_price = float(sensex.get("price", 0))
            sensex_change = float(sensex.get("change_pct", 0))
            diff_vs_sensex = round(sc - sensex_change, 2)
            requested_stock = {
                "symbol": requested.get("symbol"),
                "price": sp,
                "change_pct": sc,
                "vs_sensex_pct": diff_vs_sensex,
                "outperforming": diff_vs_sensex > 0,
            }
            sensex_comparison = {
                "sensex_level": sensex_price,
                "sensex_change_pct": sensex_change,
                "requested_symbol": requested.get("symbol"),
                "requested_price": sp,
                "requested_change_pct": sc,
                "outperforming": diff_vs_sensex > 0,
                "difference_pct": diff_vs_sensex,
            }

        # Data for prompt: top 10 only for speed
        data_str = "\n".join(str(d) for d in stock_data[:10] if isinstance(d, dict))
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Stock Agent. Be concise (1-2 sentences max).
- Specific stock asked: state price, %% change, vs Sensex in one line.
- No stock found: say so briefly and list available symbols.
- General: one-line market summary."""),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n" if req else "User: overview\n") + "Answer in 1-2 sentences."),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"data": data_str, "requirement": req or "Overview."})

        rising = [s for s in stock_data if isinstance(s, dict) and not s.get("is_index") and s.get("change_pct", 0) > 0]
        falling = [s for s in stock_data if isinstance(s, dict) and not s.get("is_index") and s.get("change_pct", 0) < 0]
        # When user asked for a particular stock: show only that stock (and Sensex for context)
        if requested:
            chart_data = [{"symbol": requested.get("symbol", "?"), "change_pct": float(requested.get("change_pct", 0)), "price": float(requested.get("price", 0))}]
            if sensex and requested.get("symbol") != sensex.get("symbol"):
                chart_data.insert(0, {"symbol": sensex.get("symbol", "?"), "change_pct": float(sensex.get("change_pct", 0)), "price": float(sensex.get("price", 0))})
            rising, falling = [], []
        else:
            chart_data = [
                {"symbol": s.get("symbol", "?"), "change_pct": float(s.get("change_pct", 0)), "price": float(s.get("price", 0))}
                for s in stock_data[:18] if isinstance(s, dict)
            ]

        all_stocks = [
            {"symbol": s.get("symbol", "?"), "price": float(s.get("price", 0)), "change_pct": float(s.get("change_pct", 0)), "volume": int(s.get("volume", 0)), "name": s.get("name")}
            for s in stock_data if isinstance(s, dict)
        ]

        # ── ML forecast for the requested symbol (or Sensex by default) ──
        forecast_target = (requested or sensex or (stock_data[0] if stock_data else None))
        forecast_block = _ml_forecast(
            (forecast_target or {}).get("symbol", "") if isinstance(forecast_target, dict) else "",
            steps=5,
        ) if forecast_target else None

        result = {
            "agent": self.agent_id,
            "summary": response if isinstance(response, str) else getattr(response, "content", str(response)),
            "alerts": [],
            "rising": rising[:10],
            "falling": falling[:10],
            "chart_data": chart_data,
            "all_stocks": all_stocks,
            "requested_stock": requested_stock,
            "sensex_comparison": sensex_comparison,
            "forecast": forecast_block,
        }
        self.remember(result["summary"], {"type": "stock_summary"})
        return result

    def stream_run(self, context: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield meta (chart, rising, falling) first for instant UI, then stream LLM summary chunks."""
        stock_data = context.get("stock_data", [])
        if not stock_data:
            yield {"done": {
                "agent": self.agent_id,
                "summary": "No stock data available. Run the stock pipeline to sync market data.",
                "alerts": [], "rising": [], "falling": [], "chart_data": [], "all_stocks": [],
                "requested_stock": None, "sensex_comparison": None,
            }}
            return

        req = (context.get("user_requirement") or "").strip()
        sensex = _get_sensex(stock_data)
        requested = _find_requested_stock(stock_data, req) if req else None
        requested_stock = None
        sensex_comparison = None
        if requested and sensex:
            sp = float(requested.get("price", 0))
            sc = float(requested.get("change_pct", 0))
            sensex_price = float(sensex.get("price", 0))
            sensex_change = float(sensex.get("change_pct", 0))
            diff_vs_sensex = round(sc - sensex_change, 2)
            requested_stock = {
                "symbol": requested.get("symbol"),
                "price": sp,
                "change_pct": sc,
                "vs_sensex_pct": diff_vs_sensex,
                "outperforming": diff_vs_sensex > 0,
            }
            sensex_comparison = {
                "sensex_level": sensex_price,
                "sensex_change_pct": sensex_change,
                "requested_symbol": requested.get("symbol"),
                "requested_price": sp,
                "requested_change_pct": sc,
                "outperforming": diff_vs_sensex > 0,
                "difference_pct": diff_vs_sensex,
            }
        rising = [s for s in stock_data if isinstance(s, dict) and not s.get("is_index") and s.get("change_pct", 0) > 0][:10]
        falling = [s for s in stock_data if isinstance(s, dict) and not s.get("is_index") and s.get("change_pct", 0) < 0][:10]
        if requested:
            chart_data = [{"symbol": requested.get("symbol", "?"), "change_pct": float(requested.get("change_pct", 0)), "price": float(requested.get("price", 0))}]
            if sensex and requested.get("symbol") != sensex.get("symbol"):
                chart_data.insert(0, {"symbol": sensex.get("symbol", "?"), "change_pct": float(sensex.get("change_pct", 0)), "price": float(sensex.get("price", 0))})
            rising, falling = [], []
        else:
            chart_data = [
                {"symbol": s.get("symbol", "?"), "change_pct": float(s.get("change_pct", 0)), "price": float(s.get("price", 0))}
                for s in stock_data[:18] if isinstance(s, dict)
            ]
        # Send meta immediately so UI can show chart and structure
        forecast_target = (requested or sensex or (stock_data[0] if stock_data else None))
        forecast_block = _ml_forecast(
            (forecast_target or {}).get("symbol", "") if isinstance(forecast_target, dict) else "",
            steps=5,
        ) if forecast_target else None

        yield {"meta": {
            "agent": self.agent_id,
            "chart_data": chart_data,
            "rising": rising,
            "falling": falling,
            "requested_stock": requested_stock,
            "sensex_comparison": sensex_comparison,
            "alerts": [],
            "forecast": forecast_block,
        }}
        data_str = "\n".join(str(d) for d in stock_data[:10] if isinstance(d, dict))
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Stock Agent. Be concise (1-2 sentences max).
- Specific stock asked: state price, %% change, vs Sensex in one line.
- No stock found: say so briefly and list available symbols.
- General: one-line market summary."""),
            ("human", "Data:\n{data}\n\n" + ("User: {requirement}\n" if req else "User: overview\n") + "Answer in 1-2 sentences."),
        ])
        chain = prompt | self.llm
        full_summary_parts = []
        for chunk in chain.stream({"data": data_str, "requirement": req or "Overview."}):
            text = chunk.content if hasattr(chunk, "content") and chunk.content else ""
            if text:
                full_summary_parts.append(text)
                yield {"text": text}
        summary = "".join(full_summary_parts)
        self.remember(summary, {"type": "stock_summary"})
        all_stocks = [
            {"symbol": s.get("symbol", "?"), "price": float(s.get("price", 0)), "change_pct": float(s.get("change_pct", 0)), "volume": int(s.get("volume", 0)), "name": s.get("name")}
            for s in stock_data if isinstance(s, dict)
        ]
        yield {"done": {
            "agent": self.agent_id,
            "summary": summary,
            "alerts": [],
            "rising": rising,
            "falling": falling,
            "chart_data": chart_data,
            "all_stocks": all_stocks,
            "requested_stock": requested_stock,
            "sensex_comparison": sensex_comparison,
            "forecast": forecast_block,
        }}
