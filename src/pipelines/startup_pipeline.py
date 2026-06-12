"""Startup intelligence sync — new registrations, founders, sectors, funding (offline)."""
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


def run_startup_sync() -> list[dict[str, Any]]:
    """Sync startup/new business data from configured path."""
    settings = get_settings()
    base = Path(settings.startup_data_path)
    out: list[dict[str, Any]] = []

    for name in ("startups.json", "registrations.json", "new_entities.json"):
        out = _load_json_or_empty(base / name)
        if out:
            break

    return out


def get_startup_data() -> list[dict[str, Any]]:
    return run_startup_sync()
