from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - useful before dependencies are installed
    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Received: {raw_value}") from exc


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number. Received: {raw_value}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed application settings loaded from environment variables."""

    secret_key: str
    flask_env: str
    ollama_base_url: str
    ollama_llm_model: str
    ollama_embedding_model: str
    ollama_temperature: float
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_metric: str
    auto_create_index: bool
    retriever_top_k: int
    chunk_size: int
    chunk_overlap: int
    max_chat_history_messages: int
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from .env and the current shell environment."""
        load_dotenv(PROJECT_ROOT / ".env")

        return cls(
            secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
            flask_env=os.getenv("FLASK_ENV", "development"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_llm_model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            ollama_temperature=_get_float("OLLAMA_TEMPERATURE", 0.2),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ai-engineering-chatbot-768"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "ai-engineering"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_metric=os.getenv("PINECONE_METRIC", "cosine"),
            auto_create_index=_get_bool("AUTO_CREATE_INDEX", True),
            retriever_top_k=_get_int("RETRIEVER_TOP_K", 4),
            chunk_size=_get_int("CHUNK_SIZE", 900),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
            max_chat_history_messages=_get_int("MAX_CHAT_HISTORY_MESSAGES", 8),
            data_dir=PROJECT_ROOT / "data" / "raw",
        )

    @property
    def pinecone_is_configured(self) -> bool:
        """Return True when a real Pinecone API key is present."""
        return bool(self.pinecone_api_key and "replace-with" not in self.pinecone_api_key)
