from __future__ import annotations

from typing import Any


def init_normal(tensor: Any, std: float = 0.02) -> Any:  # pragma: no cover - helper stub
    try:
        import torch  # type: ignore

        if isinstance(tensor, torch.Tensor):
            return torch.nn.init.trunc_normal_(tensor, std=std)
    except Exception:
        pass
    return tensor


__all__ = ["init_normal"]
