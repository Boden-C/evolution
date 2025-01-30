from __future__ import annotations

from typing import Dict, Mapping


def check_safetensors_metadata(meta: Dict[str, str]) -> bool:
    return bool(meta)


def safetensors_file_to_state_dict(path: str) -> Mapping[str, object]:  # pragma: no cover - I/O shim
    try:
        from safetensors.torch import load_file  # type: ignore

        return load_file(path)
    except Exception:
        # Fallback minimal behavior
        return {}


__all__ = ["check_safetensors_metadata", "safetensors_file_to_state_dict"]
