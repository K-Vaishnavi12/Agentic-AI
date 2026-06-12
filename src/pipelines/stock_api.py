"""
Optional: fetch live Sensex, Nifty, and stock data from Yahoo Finance (no API key).
Run this when you have internet to update data/stock/stock.json; the app stays offline at runtime.
"""
from pathlib import Path
from typing import Any

from src.config import get_settings


# Yahoo Finance symbols: indices and NSE stocks (.NS = NSE)
SENSEX_SYMBOL = "^BSESN"   # BSE Sensex
NIFTY_SYMBOL = "^NSEI"    # Nifty 50
DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
]


def fetch_from_yahoo() -> list[dict[str, Any]]:
    """Fetch Sensex, Nifty, and stocks from Yahoo Finance. Returns list of {symbol, price, change_pct, ...}."""
    try:
        import yfinance as yf
    except ImportError:
        return []

    out: list[dict[str, Any]] = []
    symbols = [SENSEX_SYMBOL, NIFTY_SYMBOL] + DEFAULT_STOCKS

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            hist = ticker.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            else:
                price = float(info.get("regularMarketPrice") or info.get("previousClose") or 0)
                prev = price
            change_pct = round((price - prev) / prev * 100, 2) if prev and prev != 0 else 0
            display_name = sym.replace("^", "").replace(".NS", "")
            if sym == SENSEX_SYMBOL:
                display_name = "SENSEX"
            elif sym == NIFTY_SYMBOL:
                display_name = "NIFTY50"
            out.append({
                "symbol": display_name,
                "price": round(price, 2),
                "change_pct": change_pct,
                "volume": int(info.get("volume") or 0),
                "is_index": sym.startswith("^"),
            })
        except Exception:
            continue

    return out


def run_stock_api_sync() -> list[dict[str, Any]]:
    """
    Fetch live data from Yahoo Finance and write to data/stock/stock.json.
    Returns the fetched data; on failure returns empty list (caller can keep using existing JSON).
    """
    data = fetch_from_yahoo()
    if not data:
        return []

    settings = get_settings()
    base = Path(settings.stock_data_path)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "stock.json"

    # Ensure float for price/change_pct for JSON
    for row in data:
        row["price"] = float(row.get("price", 0))
        row["change_pct"] = float(row.get("change_pct", 0))

    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    result = run_stock_api_sync()
    print(f"Updated {len(result)} symbols in data/stock/stock.json")
    for r in result[:5]:
        print(" ", r.get("symbol"), r.get("price"), r.get("change_pct"), "%")
