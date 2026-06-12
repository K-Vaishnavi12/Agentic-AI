"""Stock data sync pipeline — reads from local/offline sources."""
import json
from pathlib import Path
from typing import Any
from src.config import get_settings


def _load_json_or_empty(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def run_stock_sync() -> list[dict[str, Any]]:
    """Sync stock/market data from configured path. Offline: use local JSON/CSV."""
    settings = get_settings()
    base = Path(settings.stock_data_path)
    out: list[dict[str, Any]] = []

    # Prefer stock.json (list of {symbol, price, change_pct, ...})
    for name in ("stock.json", "market.json", "stocks.json"):
        out = _load_json_or_empty(base / name)
        if out:
            break

    # CSV fallback
    for csv_name in ("stock.csv", "market.csv"):
        csv_path = base / csv_name
        if csv_path.exists():
            import csv
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = dict(row)
                    if "change_pct" in r:
                        try:
                            r["change_pct"] = float(r["change_pct"])
                        except ValueError:
                            r["change_pct"] = 0.0
                    out.append(r)
            break
    return out


def get_stock_data() -> list[dict[str, Any]]:
    """Return latest stock data (after sync)."""
    return run_stock_sync()
