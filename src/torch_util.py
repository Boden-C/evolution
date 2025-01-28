from __future__ import annotations

from typing import Optional


def get_default_device(prefer_cuda: bool = False) -> str:
    try:
        import torch  # type: ignore

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def seed_all(seed: int) -> None:
    try:
        import random
        import os
        import numpy as np  # type: ignore
        import torch  # type: ignore

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Keep side-effect free when torch/numpy not installed
        pass


def move_to_device(x, device: Optional[str] = None):  # pragma: no cover - runtime convenience
    if device is None:
        device = get_default_device()
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.to(device)
    except Exception:
        pass
    return x


__all__ = ["get_default_device", "seed_all", "move_to_device"]
