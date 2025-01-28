from __future__ import annotations

import enum
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


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

__all__ = ["StrEnum", "dataclass_asdict_shallow"]
