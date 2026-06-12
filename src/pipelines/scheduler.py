"""Scheduled data sync — run pipelines at intervals (offline-first)."""
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from apscheduler.schedulers.background import BackgroundScheduler
from src.config import get_settings
from .stock_pipeline import run_stock_sync
from .legal_pipeline import run_legal_sync
from .startup_pipeline import run_startup_sync

# In-memory cache of latest sync (no cloud)
_cached_context: dict[str, Any] = {}


def _run_all_syncs() -> None:
    global _cached_context
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_stock = executor.submit(run_stock_sync)
        f_legal = executor.submit(run_legal_sync)
        f_startup = executor.submit(run_startup_sync)
        _cached_context = {
            "stock_data": f_stock.result(),
            "legal_data": f_legal.result(),
            "startup_data": f_startup.result(),
            "shareholder_data": [],  # Extend with shareholder pipeline if needed
        }


def get_context_from_pipelines() -> dict[str, Any]:
    """Return latest synced context for agents. Call _run_all_syncs or start_scheduler first."""
    if not _cached_context:
        _run_all_syncs()
    return _cached_context.copy()


def start_scheduler(interval_minutes: int = 60) -> BackgroundScheduler:
    """Start background scheduler to sync data at interval. Offline-friendly."""
    _run_all_syncs()
    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_all_syncs, "interval", minutes=interval_minutes)
    scheduler.start()
    return scheduler
