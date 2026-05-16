from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from dotenv import load_dotenv


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Check local Ollama chat and embedding models.")
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--llm", default=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct"))
    parser.add_argument(
        "--embedding",
        default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )
    args = parser.parse_args()

    print(f"Checking Ollama at {args.base_url}")

    tags_response = requests.get(f"{args.base_url}/api/tags", timeout=30)
    tags_response.raise_for_status()
    installed_models = [model["name"] for model in tags_response.json().get("models", [])]
    print("Installed models:", ", ".join(installed_models) or "none")

    chat_payload = {
        "model": args.llm,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "Reply with one short sentence explaining what RAG means.",
            }
        ],
    }
    chat_response = _post_json(f"{args.base_url}/api/chat", chat_payload)
    print("\nChat model response:")
    print(chat_response.get("message", {}).get("content", "").strip())

    embed_payload = {
        "model": args.embedding,
        "input": "AI engineering chatbots can retrieve relevant technical snippets before answering.",
    }
    embed_response = _post_json(f"{args.base_url}/api/embed", embed_payload)
    embeddings = embed_response.get("embeddings") or []
    dimension = len(embeddings[0]) if embeddings else 0
    print(f"\nEmbedding model dimension: {dimension}")

    if not dimension:
        raise RuntimeError("Ollama did not return an embedding vector.")

    print("\nOllama check passed.")


if __name__ == "__main__":
    main()
