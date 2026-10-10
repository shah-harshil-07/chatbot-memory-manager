"""Application settings, loaded from environment variables or a .env file."""
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: SecretStr
    openai_chat_model: str
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = Field(1536, gt=0)
    openai_timeout_s: float = 30.0
    openai_max_retries: int = 3  # SDK retries 429/5xx/timeouts with exponential backoff

    # --- Qdrant -------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "agent_memories"

    # --- Retrieval ----------------------------------------------------------
    recall_top_k: int = 5            # memories injected into each prompt
    recall_min_score: float = 0.30   # ignore weakly related memories
    history_turns: int = 10          # recent chat messages sent verbatim

    # --- Write path (reconciliation) ---------------------------------------
    reconcile_top_k: int = 5         # existing memories shown to the LLM when deciding ADD/UPDATE/...
    duplicate_threshold: float = 0.95  # above this similarity -> NOOP without calling the LLM

    # --- Maintenance --------------------------------------------------------
    consolidation_threshold: float = 0.88  # similarity that counts as "near-duplicate"
    max_cluster_size: int = 20             # larger clusters are skipped (likely chaining)
    purge_after_days: int = 30             # hard-delete invalidated memories after this

    # --- HTTP API / logging -------------------------------------------------
    app_api_key: SecretStr | None = None
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
