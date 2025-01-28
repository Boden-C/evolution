from __future__ import annotations

from dataclasses import dataclass

from ..aliases import PathOrStr


@dataclass
class Checkpointer:
    save_folder: PathOrStr

    def save(self, name: str = "model") -> None:  # pragma: no cover - I/O shim
        return None

    def load(self, name: str = "model") -> dict | None:  # pragma: no cover - I/O shim
        return None


__all__ = ["Checkpointer"]
