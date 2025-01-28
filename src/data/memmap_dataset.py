from __future__ import annotations

from typing import Iterator, Sequence


class MemmapDataset:
    def __init__(self, path: str):
        self.path = path

    def __iter__(self) -> Iterator[bytes]:  # pragma: no cover - placeholder
        return iter(())


__all__ = ["MemmapDataset"]
