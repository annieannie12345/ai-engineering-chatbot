from __future__ import annotations

import time
from typing import Any

from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings
from app.rag.errors import ConfigurationError


def get_embeddings(settings: Settings) -> OllamaEmbeddings:
    """Create the local embedding model used for both ingestion and retrieval."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_embedding_dimension(embeddings: OllamaEmbeddings) -> int:
    """Probe Ollama once so the Pinecone index dimension matches the embedding model."""
    probe_vector = embeddings.embed_query("embedding dimension probe")
    return len(probe_vector)


def get_pinecone_client(settings: Settings) -> Pinecone:
    """Create a Pinecone client after validating that credentials are configured."""
    if not settings.pinecone_is_configured:
        raise ConfigurationError(
            "PINECONE_API_KEY is missing. Add it to your .env file before ingesting or chatting."
        )
    return Pinecone(api_key=settings.pinecone_api_key)


def _has_index(client: Pinecone, index_name: str) -> bool:
    if hasattr(client, "has_index"):
        return bool(client.has_index(index_name))

    indexes = client.list_indexes()
    if hasattr(indexes, "names"):
        return index_name in indexes.names()
    return index_name in [index.get("name") for index in indexes]


def _get_index_dimension(description: Any) -> int | None:
    if hasattr(description, "dimension"):
        return description.dimension
    if isinstance(description, dict):
        return description.get("dimension")
    return None


def _index_is_ready(description: Any) -> bool:
    status = getattr(description, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    if status is not None and hasattr(status, "ready"):
        return bool(status.ready)
    return True


def ensure_pinecone_index(settings: Settings, embeddings: OllamaEmbeddings | None = None):
    """Create the Pinecone index when needed and validate its vector dimension."""
    client = get_pinecone_client(settings)
    resolved_embeddings = embeddings or get_embeddings(settings)
    dimension = get_embedding_dimension(resolved_embeddings)

    if not _has_index(client, settings.pinecone_index_name):
        client.create_index(
            name=settings.pinecone_index_name,
            vector_type="dense",
            dimension=dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
            deletion_protection="disabled",
            tags={"project": "ai-engineering-chatbot", "environment": settings.flask_env},
        )

    for _ in range(90):
        description = client.describe_index(settings.pinecone_index_name)
        existing_dimension = _get_index_dimension(description)
        if existing_dimension and existing_dimension != dimension:
            raise ConfigurationError(
                "Pinecone index dimension mismatch. "
                f"Index has {existing_dimension}, but {settings.ollama_embedding_model} "
                f"produces {dimension}. Use a new PINECONE_INDEX_NAME or recreate the index."
            )
        if _index_is_ready(description):
            return client.Index(settings.pinecone_index_name)
        time.sleep(1)

    raise TimeoutError(f"Pinecone index {settings.pinecone_index_name} was not ready in time.")


def get_vector_store(
    settings: Settings,
    embeddings: OllamaEmbeddings | None = None,
    *,
    ensure_index_exists: bool | None = None,
) -> PineconeVectorStore:
    """Return the LangChain vector store wrapper around the configured Pinecone index."""
    resolved_embeddings = embeddings or get_embeddings(settings)
    should_ensure = settings.auto_create_index if ensure_index_exists is None else ensure_index_exists

    if should_ensure:
        index = ensure_pinecone_index(settings, resolved_embeddings)
    else:
        client = get_pinecone_client(settings)
        index = client.Index(settings.pinecone_index_name)

    return PineconeVectorStore(
        index=index,
        embedding=resolved_embeddings,
        namespace=settings.pinecone_namespace,
    )
