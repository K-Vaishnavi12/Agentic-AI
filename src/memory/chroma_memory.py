"""ChromaDB vector memory for agent context — runs locally."""
from typing import List, Optional
from src.config import get_settings

try:
    from langchain_ollama import OllamaEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    from langchain_community.vectorstores import Chroma  # type: ignore


def _collection_name(agent_id: str) -> str:
    return f"agent_memory_{agent_id}"


def get_vector_store(agent_id: str):
    settings = get_settings()
    # Use nomic-embed-text for embeddings (run: ollama pull nomic-embed-text)
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model="nomic-embed-text",
    )
    return Chroma(
        collection_name=_collection_name(agent_id),
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def add_to_memory(agent_id: str, text: str, metadata: Optional[dict] = None) -> None:
    store = get_vector_store(agent_id)
    store.add_texts([text], metadatas=[metadata or {}])


def query_memory(agent_id: str, query: str, k: int = 5) -> List[str]:
    store = get_vector_store(agent_id)
    docs = store.similarity_search(query, k=k)
    return [d.page_content for d in docs]
