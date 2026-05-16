from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceSnippet:
    source: str
    excerpt: str
    page: int | None = None
    score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChatResult:
    answer: str
    sources: list[SourceSnippet]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }
