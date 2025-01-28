from __future__ import annotations

from typing import Iterable, Iterator, List


class InMemoryTextDataset:
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - simple
        return iter(self.texts)


__all__ = ["InMemoryTextDataset"]
