from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class IterableDataset:
    def __init__(self, data: Iterable[T]):
        self._data = data

    def __iter__(self) -> Iterator[T]:  # pragma: no cover - simple
        return iter(self._data)


__all__ = ["IterableDataset"]
