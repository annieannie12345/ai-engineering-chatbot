from __future__ import annotations

import json

from flask import Blueprint, Response, current_app, jsonify, render_template, request, stream_with_context

from app.rag.errors import ConfigurationError


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    settings = current_app.config["SETTINGS"]
    return render_template(
        "index.html",
        llm_model=settings.ollama_llm_model,
        embedding_model=settings.ollama_embedding_model,
        namespace=settings.pinecone_namespace,
    )


@main_bp.get("/api/health")
def health():
    settings = current_app.config["SETTINGS"]
    return jsonify(
        {
            "status": "ok",
            "llm_model": settings.ollama_llm_model,
            "embedding_model": settings.ollama_embedding_model,
            "pinecone_configured": settings.pinecone_is_configured,
            "pinecone_index": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
        }
    )


@main_bp.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", ""))
    history = payload.get("history") or []

    try:
        result = current_app.rag_service.answer(message, history)  # type: ignore[attr-defined]
        return jsonify(result.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # pragma: no cover - protects API clients from stack traces
        current_app.logger.exception("Chat request failed")
        return jsonify({"error": f"Chat request failed: {exc}"}), 500


@main_bp.post("/api/chat/stream")
def chat_stream():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", ""))
    history = payload.get("history") or []

    def generate():
        try:
            for event in current_app.rag_service.stream_answer(message, history):  # type: ignore[attr-defined]
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except ConfigurationError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:  # pragma: no cover - protects streaming clients
            current_app.logger.exception("Streaming chat request failed")
            yield f"data: {json.dumps({'type': 'error', 'error': f'Chat request failed: {exc}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
