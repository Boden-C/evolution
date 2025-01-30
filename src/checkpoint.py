from __future__ import annotations

# Re-export runtime checkpointer API and provide convenience constructors.
from .runtime.checkpoint import Checkpointer as Checkpointer  # noqa: F401
from .aliases import PathOrStr


def make_checkpointer(save_folder: PathOrStr) -> Checkpointer:
    return Checkpointer(save_folder)


__all__ = ["Checkpointer", "make_checkpointer"]
