"""Ollama-backed LLM for fully offline inference."""
from langchain_ollama import ChatOllama
from src.config import get_settings


def get_llm(temperature: float = 0.2, model: str | None = None, num_predict: int = 150):
    """Shorter num_predict reduces buffering time. num_ctx limits context window for speed."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=512,
    )
