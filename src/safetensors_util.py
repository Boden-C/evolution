from __future__ import annotations

from typing import Dict


def check_safetensors_metadata(meta: Dict[str, str]) -> bool:
    return bool(meta)


__all__ = ["check_safetensors_metadata"]
