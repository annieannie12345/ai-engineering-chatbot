from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.rag.vector_store import ensure_pinecone_index, get_embedding_dimension, get_embeddings


def main() -> None:
    settings = Settings.from_env()
    embeddings = get_embeddings(settings)
    dimension = get_embedding_dimension(embeddings)
    ensure_pinecone_index(settings, embeddings)

    print("Pinecone index is ready.")
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Namespace: {settings.pinecone_namespace}")
    print(f"Embedding model: {settings.ollama_embedding_model}")
    print(f"Vector dimension: {dimension}")


if __name__ == "__main__":
    main()
