"""Application settings — env-driven, offline-first."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ─────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    # Use a small model (e.g. phi3, tinyllama) if you have <5GB free RAM; mistral needs ~4.5GB
    ollama_model: str = Field(default="phi3", alias="OLLAMA_MODEL")

    # ── Storage ─────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    stock_data_path: str = Field(default="./data/stock", alias="STOCK_DATA_PATH")
    legal_data_path: str = Field(default="./data/legal", alias="LEGAL_DATA_PATH")
    startup_data_path: str = Field(default="./data/startup", alias="STARTUP_DATA_PATH")

    # ── RBAC ────────────────────────────────────────────────────────
    strict_rbac: bool = Field(default=True, alias="STRICT_RBAC")

    # ── ML pipeline ─────────────────────────────────────────────────
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    reports_dir: str = Field(default="./reports", alias="REPORTS_DIR")
    ml_data_dir: str = Field(default="./data/ml", alias="ML_DATA_DIR")
    ml_seed: int = Field(default=42, alias="ML_SEED")
    ml_lookback: int = Field(default=60, alias="ML_LOOKBACK")
    ml_horizon: int = Field(default=5, alias="ML_HORIZON")
    ml_transformer_base: str = Field(default="distilbert-base-uncased", alias="ML_TRANSFORMER_BASE")
    ml_skip_transformer: bool = Field(default=False, alias="ML_SKIP_TRANSFORMER")

    class Config:
        env_file = ".env"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        for p in (
            self.chroma_persist_dir,
            self.data_dir,
            self.stock_data_path,
            self.legal_data_path,
            self.startup_data_path,
            self.models_dir,
            self.reports_dir,
            self.ml_data_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
