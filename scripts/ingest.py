from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from app.config import Settings
from app.rag.document_loader import build_chunk_ids, load_documents, split_documents
from app.rag.vector_store import ensure_pinecone_index, get_embeddings, get_vector_store


def _batch_items(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AI engineering documents into Pinecone.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .md, .txt, and .pdf files.",
    )
    parser.add_argument(
        "--clear-namespace",
        action="store_true",
        help="Delete existing vectors in the configured namespace before ingestion.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = Settings.from_env()
    data_dir = args.data_dir or settings.data_dir

    raw_documents = load_documents(data_dir)
    if not raw_documents:
        raise SystemExit(f"No supported documents found in {data_dir}")

    chunks: list[Document] = split_documents(raw_documents, settings)
    chunk_ids = build_chunk_ids(chunks)

    embeddings = get_embeddings(settings)
    index = ensure_pinecone_index(settings, embeddings)

    if args.clear_namespace:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
        print(f"Cleared namespace: {settings.pinecone_namespace}")

    vector_store = get_vector_store(settings, embeddings, ensure_index_exists=False)

    for documents_batch, ids_batch in zip(
        _batch_items(chunks, args.batch_size),
        _batch_items(chunk_ids, args.batch_size),
        strict=True,
    ):
        vector_store.add_documents(documents=list(documents_batch), ids=list(ids_batch))

    print("Ingestion complete.")
    print(f"Loaded files from: {data_dir}")
    print(f"Raw documents: {len(raw_documents)}")
    print(f"Chunks upserted: {len(chunks)}")
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Namespace: {settings.pinecone_namespace}")


if __name__ == "__main__":
    main()
