from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from app.config import Settings
from app.rag.prompts import build_ai_engineering_rag_prompt
from app.rag.schemas import ChatResult, SourceSnippet
from app.rag.vector_store import get_vector_store


class AiEngineeringRagService:
    """Coordinates retrieval from Pinecone and answer generation through Ollama."""

    refusal_message = (
        "I don't have an AI engineering answer for that in the knowledge base. "
        "Try asking about ML, deep learning, RAG, LLMs, agents, evaluation, or MLOps."
    )
    greeting_message = (
        "Hi! I'm doing great and ready to help. Ask me anything about ML, deep learning, "
        "RAG, LLMs, agents, evaluation, or MLOps."
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
        )
        self.prompt = build_ai_engineering_rag_prompt()

    def answer(self, question: str, history: list[dict[str, Any]] | None = None) -> ChatResult:
        """Return a complete RAG answer for API clients that do not need streaming."""
        cleaned_question = self._clean_question(question)
        if self._is_small_talk(cleaned_question):
            return ChatResult(answer=self.greeting_message, sources=[])
        if self._should_refuse_without_retrieval(cleaned_question):
            return ChatResult(answer=self.refusal_message, sources=[])

        retrieval_query = self._build_retrieval_query(cleaned_question)
        docs_with_scores = self._retrieve(retrieval_query)
        messages = self._build_prompt_messages(
            cleaned_question,
            history if self._is_contextual_follow_up(cleaned_question) else [],
            docs_with_scores,
        )
        response = self.llm.invoke(messages)

        return ChatResult(
            answer=str(response.content),
            sources=self._build_sources(docs_with_scores),
        )

    def stream_answer(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield answer tokens first, then sources, for a responsive chat UI."""
        cleaned_question = self._clean_question(question)
        if self._is_small_talk(cleaned_question):
            yield {"type": "token", "content": self.greeting_message}
            yield {"type": "sources", "sources": []}
            yield {"type": "done"}
            return

        if self._should_refuse_without_retrieval(cleaned_question):
            yield {"type": "token", "content": self.refusal_message}
            yield {"type": "sources", "sources": []}
            yield {"type": "done"}
            return

        retrieval_query = self._build_retrieval_query(cleaned_question)
        docs_with_scores = self._retrieve(retrieval_query)
        messages = self._build_prompt_messages(
            cleaned_question,
            history if self._is_contextual_follow_up(cleaned_question) else [],
            docs_with_scores,
        )

        for chunk in self.llm.stream(messages):
            token = getattr(chunk, "content", "")
            if token:
                yield {"type": "token", "content": str(token)}

        yield {
            "type": "sources",
            "sources": [source.to_dict() for source in self._build_sources(docs_with_scores)],
        }
        yield {"type": "done"}

    def _retrieve(self, question: str):
        vector_store = get_vector_store(self.settings)
        return vector_store.similarity_search_with_score(question, k=self.settings.retriever_top_k)

    def _build_prompt_messages(
        self,
        question: str,
        history: list[dict[str, Any]] | None,
        docs_with_scores,
    ) -> list[BaseMessage]:
        context = self._format_context(docs_with_scores)
        prompt_value = self.prompt.invoke(
            {
                "question": question,
                "context": context,
                "chat_history": self._format_chat_history(history or []),
            }
        )
        return prompt_value.to_messages()

    def _format_chat_history(self, history: list[dict[str, Any]]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        limited_history = history[-self.settings.max_chat_history_messages :]

        for item in limited_history:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    @staticmethod
    def _format_context(docs_with_scores) -> str:
        if not docs_with_scores:
            return "No relevant context was found in the vector database."

        formatted_chunks: list[str] = []
        for position, (document, score) in enumerate(docs_with_scores, start=1):
            source = document.metadata.get("file_name") or document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            location = f"{source}, page {page}" if page is not None else str(source)
            formatted_chunks.append(
                f"[Source {position}: {location}; score={score:.4f}]\n{document.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    @staticmethod
    def _build_sources(docs_with_scores) -> list[SourceSnippet]:
        sources: list[SourceSnippet] = []
        for document, score in docs_with_scores:
            excerpt = " ".join(document.page_content.split())
            if len(excerpt) > 260:
                excerpt = excerpt[:257].rstrip() + "..."
            page = document.metadata.get("page")
            sources.append(
                SourceSnippet(
                    source=str(document.metadata.get("file_name") or document.metadata.get("source")),
                    page=int(page) if isinstance(page, int) else None,
                    score=float(score) if score is not None else None,
                    excerpt=excerpt,
                )
            )
        return sources

    @staticmethod
    def _clean_question(question: str) -> str:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Message cannot be empty.")
        if len(cleaned_question) > 4000:
            raise ValueError("Message is too long. Keep it under 4000 characters.")
        return cleaned_question

    @classmethod
    def _should_refuse_without_retrieval(cls, question: str) -> bool:
        """Catch off-domain input before irrelevant retrieval or chat history can distract the LLM."""
        normalized = question.strip().lower()
        if len(normalized) < 3:
            return True

        if cls._is_small_talk(normalized):
            return False

        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", normalized)
        if not tokens:
            return True

        if cls._is_personal_statement(normalized):
            return True

        compact = "".join(tokens)
        if cls._is_alphabet_sequence(compact):
            return True

        if len(tokens) == 1:
            token = tokens[0]
            if token in cls._known_short_ai_terms():
                return False
            if len(token) >= 5 and cls._looks_like_random_token(token):
                return True

        return not cls._contains_ai_domain_signal(normalized, tokens)

    @classmethod
    def _build_retrieval_query(cls, question: str) -> str:
        """Blend semantic meaning with domain keywords so short queries retrieve better context."""
        normalized = question.lower()
        expansions: list[str] = []

        for term, expansion in cls._query_expansions().items():
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                expansions.append(expansion)

        if not expansions:
            return question

        return f"{question}\n\nRelated AI engineering keywords: {'; '.join(expansions)}"

    @classmethod
    def _is_contextual_follow_up(cls, question: str) -> bool:
        """Use prior chat only for true follow-up wording, not for every short message."""
        normalized = question.strip().lower()
        follow_up_starters = (
            "also",
            "and ",
            "compare",
            "continue",
            "explain more",
            "give example",
            "how about",
            "in detail",
            "more",
            "tell me more",
            "what about",
            "why",
        )
        pronoun_references = {"it", "that", "this", "they", "them", "those"}
        tokens = set(re.findall(r"[a-zA-Z]+", normalized))

        return normalized.startswith(follow_up_starters) or bool(tokens & pronoun_references)

    @staticmethod
    def _is_alphabet_sequence(text: str) -> bool:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return len(text) >= 5 and text in alphabet

    @staticmethod
    def _looks_like_random_token(token: str) -> bool:
        if any(character.isdigit() for character in token):
            return False
        common_ai_fragments = {
            "agent",
            "attention",
            "bert",
            "chunk",
            "deep",
            "embed",
            "eval",
            "fine",
            "gradient",
            "langchain",
            "learning",
            "llm",
            "loss",
            "mlops",
            "model",
            "neural",
            "pinecone",
            "prompt",
            "rag",
            "token",
            "train",
            "transformer",
            "vector",
        }
        if any(fragment in token for fragment in common_ai_fragments):
            return False

        vowels = sum(1 for character in token if character in "aeiou")
        vowel_ratio = vowels / max(len(token), 1)
        repeated_unique_ratio = len(set(token)) / max(len(token), 1)
        return vowel_ratio < 0.2 or repeated_unique_ratio > 0.85

    @classmethod
    def _is_small_talk(cls, question: str) -> bool:
        normalized = question.strip().lower().strip("!.?")
        tokens = re.findall(r"[a-zA-Z]+", normalized)
        if normalized in cls._small_talk_terms():
            return True
        if len(tokens) <= 5 and any(phrase in normalized for phrase in cls._small_talk_phrases()):
            return True
        return False

    @staticmethod
    def _is_personal_statement(normalized: str) -> bool:
        if normalized.startswith(("my name is", "i am ", "i'm ", "call me ")):
            return True
        return False

    @classmethod
    def _contains_ai_domain_signal(cls, normalized: str, tokens: list[str]) -> bool:
        token_set = set(tokens)
        if token_set & cls._known_short_ai_terms():
            return True
        if token_set & cls._ai_domain_keywords():
            return True
        return any(phrase in normalized for phrase in cls._ai_domain_phrases())

    @staticmethod
    def _small_talk_terms() -> set[str]:
        return {
            "bye",
            "goodbye",
            "hello",
            "hey",
            "hi",
            "hii",
            "hiii",
            "howdy",
            "ok",
            "okay",
            "thanks",
            "thank",
            "yo",
        }

    @staticmethod
    def _small_talk_phrases() -> set[str]:
        return {
            "how are you",
            "how r you",
            "how are u",
            "what's up",
            "whats up",
        }

    @staticmethod
    def _ai_domain_keywords() -> set[str]:
        return {
            "agent",
            "agents",
            "ai",
            "attention",
            "backpropagation",
            "bias",
            "classification",
            "classifier",
            "cluster",
            "clustering",
            "cnn",
            "context",
            "dataset",
            "deep",
            "dl",
            "embedding",
            "embeddings",
            "evaluation",
            "fine-tuning",
            "finetuning",
            "gan",
            "gradient",
            "inference",
            "langchain",
            "learning",
            "llm",
            "llms",
            "loss",
            "machine",
            "ml",
            "mlops",
            "model",
            "models",
            "neural",
            "nlp",
            "overfitting",
            "pinecone",
            "prompt",
            "prompts",
            "rag",
            "regression",
            "retrieval",
            "rl",
            "rnn",
            "token",
            "tokenization",
            "training",
            "transformer",
            "transformers",
            "underfitting",
            "vector",
            "vectors",
            "variance",
        }

    @staticmethod
    def _ai_domain_phrases() -> set[str]:
        return {
            "artificial intelligence",
            "cross validation",
            "deep learning",
            "fine tuning",
            "large language model",
            "machine learning",
            "neural network",
            "prompt engineering",
            "retrieval augmented generation",
            "support vector machine",
            "vector database",
        }

    @staticmethod
    def _query_expansions() -> dict[str, str]:
        return {
            "ai": "artificial intelligence machine learning deep learning AI systems",
            "ann": "artificial neural network neural networks deep learning",
            "cnn": "convolutional neural network image deep learning architecture",
            "dl": "deep learning neural networks backpropagation training",
            "gan": "generative adversarial network generator discriminator deep learning",
            "llm": "large language model transformer prompting tokenization inference",
            "llms": "large language models transformers prompting tokenization inference",
            "ml": "machine learning supervised unsupervised regression classification evaluation",
            "mlops": "MLOps model deployment monitoring pipelines model registry",
            "nlp": "natural language processing tokenization embeddings transformers language models",
            "rag": "Retrieval-Augmented Generation embeddings vector database retrieval chunking grounding",
            "rl": "reinforcement learning agent policy reward environment",
            "rnn": "recurrent neural network sequence modeling deep learning",
            "svm": "support vector machine supervised learning classifier hyperplane margin kernel",
            "vae": "variational autoencoder generative model latent representation",
        }

    @staticmethod
    def _known_short_ai_terms() -> set[str]:
        return {
            "ai",
            "ann",
            "bert",
            "cnn",
            "dl",
            "gan",
            "gpt",
            "gru",
            "llm",
            "ml",
            "nlp",
            "rag",
            "rl",
            "rnn",
            "svd",
            "svm",
            "vae",
        }
