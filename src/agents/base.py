"""Base agent with LLM and optional ChromaDB memory."""
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional
from src.llm import get_llm
from src.memory import add_to_memory, query_memory


class BaseAgent(ABC):
    """Specialized agent with a dedicated role and data access."""

    agent_id: str = "base"
    name: str = "Base Agent"

    def __init__(self, use_memory: bool = True):
        self.llm = get_llm()
        self.use_memory = use_memory

    def remember(self, text: str, metadata: Optional[dict] = None) -> None:
        if self.use_memory:
            add_to_memory(self.agent_id, text, metadata)

    def recall(self, query: str, k: int = 5) -> list[str]:
        if not self.use_memory:
            return []
        return query_memory(self.agent_id, query, k=k)

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute agent logic; returns a result dict (e.g. summary, alerts, data)."""
        pass

    def stream_run(self, context: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream run: yield meta/data first, then {"text": chunk} for LLM stream, then {"done": result}. Override for real streaming."""
        result = self.run(context)
        yield {"done": result}
