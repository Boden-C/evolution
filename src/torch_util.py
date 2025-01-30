from __future__ import annotations

from typing import Any, Optional


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


def is_distributed() -> bool:
    try:
        import torch.distributed as dist  # type: ignore

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_world_size() -> int:
    if not is_distributed():
        return 1
    import torch.distributed as dist  # type: ignore

    return dist.get_world_size()


def get_global_rank() -> int:
    if not is_distributed():
        return 0
    import torch.distributed as dist  # type: ignore

    return dist.get_rank()


def barrier() -> None:  # pragma: no cover - no-op in single process
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def synchronize_value(value: float | int, op: str = "mean") -> Any:
    """Synchronize a scalar across processes. Returns the reduced value.

    Supported ops: 'mean', 'sum', 'max', 'min'. No-op if not distributed.
    """
    if not is_distributed():
        return value
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    t = torch.tensor([float(value)], device="cuda" if torch.cuda.is_available() else "cpu")
    if op == "mean":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    elif op == "sum":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    else:
        raise ValueError("Unsupported op for synchronize_value")
    return t.item()


__all__ = [
    "get_default_device",
    "seed_all",
    "move_to_device",
    "is_distributed",
    "get_world_size",
    "get_global_rank",
    "barrier",
    "synchronize_value",
]


__all__ = ["get_default_device", "seed_all", "move_to_device"]
