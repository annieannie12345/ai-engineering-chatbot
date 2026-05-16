from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def iter_source_files(data_dir: Path) -> Iterable[Path]:
    """Yield supported files from the data directory in a stable order."""
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(data_dir: Path) -> list[Document]:
    """Load Markdown, text, and PDF files into LangChain Document objects."""
    documents: list[Document] = []

    for path in iter_source_files(data_dir):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        for document in loader.load():
            document.metadata["source"] = str(path)
            document.metadata["file_name"] = path.name
            document.metadata["file_type"] = suffix.lstrip(".")
            documents.append(document)

    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into retrieval-sized chunks before embedding them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata = _clean_metadata(chunk.metadata)

    return chunks


def build_chunk_id(document: Document) -> str:
    """Create a stable ID so re-ingestion updates existing vectors instead of duplicating them."""
    source = document.metadata.get("source", "unknown-source")
    page = document.metadata.get("page", "")
    chunk_index = document.metadata.get("chunk_index", "")
    fingerprint = hashlib.sha256(
        f"{source}|{page}|{chunk_index}|{document.page_content}".encode("utf-8")
    ).hexdigest()[:32]
    return f"chunk-{fingerprint}"


def build_chunk_ids(documents: list[Document]) -> list[str]:
    return [build_chunk_id(document) for document in documents]


def _clean_metadata(metadata: dict) -> dict:
    """Keep metadata within Pinecone's primitive value requirements."""
    cleaned: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned
