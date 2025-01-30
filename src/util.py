from __future__ import annotations

import enum
from dataclasses import asdict, is_dataclass
import os
from typing import Any, Dict, Iterable, Tuple


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


def dataclass_asdict_shallow(obj: Any) -> Dict[str, Any]:
    """Return a shallow asdict for dataclasses, else wrap into a dict."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def find_latest_checkpoint(folder: str, prefix: str = "model") -> Tuple[str | None, int | None]:
    """Return the latest checkpoint directory matching prefix, with step number.

    We assume checkpoints are saved as subfolders under folder, like 'model', 'model-10', etc.
    """
    if not os.path.isdir(folder):
        return None, None
    best_step = -1
    best_path: str | None = None
    for name in os.listdir(folder):
        if not name.startswith(prefix):
            continue
        step = 0
        if "-" in name:
            try:
                step = int(name.split("-")[-1])
            except Exception:
                step = 0
        path = os.path.join(folder, name)
        if os.path.isdir(path) and step >= best_step:
            best_step = step
            best_path = path
    return best_path, (best_step if best_step >= 0 else None)


__all__ = [
    "StrEnum",
    "dataclass_asdict_shallow",
    "flatten_dict",
    "find_latest_checkpoint",
]
