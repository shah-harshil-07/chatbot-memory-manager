"""Wires the components together (simple dependency injection)."""
from dataclasses import dataclass

from qdrant_client import QdrantClient

from .agent import MemoryAgent
from .config import Settings
from .llm import LLMClient, OpenAIClient
from .maintenance import Maintenance
from .memory_manager import MemoryManager
from .store import MemoryStore


@dataclass
class Container:
    settings: Settings
    store: MemoryStore
    llm: LLMClient
    manager: MemoryManager
    agent: MemoryAgent
    maintenance: Maintenance


def build_container(settings: Settings, *, llm: LLMClient | None = None,
                    qdrant: QdrantClient | None = None) -> Container:
    qdrant = qdrant or QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=10,
    )
    store = MemoryStore(qdrant, settings.qdrant_collection, settings.embedding_dim)
    store.ensure_collection()
    llm = llm or OpenAIClient(settings)
    manager = MemoryManager(store, llm, settings)
    return Container(settings, store, llm, manager,
                     MemoryAgent(store, llm, manager, settings),
                     Maintenance(store, llm, settings))
