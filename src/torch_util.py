"""Torch utility functions for distributed and device operations.

Provides enterprise-level best practices, logging, and robust type checks.
"""

from __future__ import annotations

import gc
from .logger import get_logger
import os
import random
from typing import Any, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist

T = TypeVar("T")
V = TypeVar("V", bool, int, float)

logger = get_logger(__name__)

# Default devices


def get_default_device(prefer_cuda: bool = False) -> torch.device:
    """Return default torch device, optionally preferring CUDA."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    """Seed random, numpy, and torch generators. Seed must be in [0, 2**32-1]."""
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(f"Seed {seed} is invalid; must be between 0 and 2**32-1")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.debug("Random generators seeded with %d", seed)


def is_distributed() -> bool:
    """Return True if torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return number of processes in distributed group, or 1."""
    return dist.get_world_size() if is_distributed() else 1


def get_global_rank() -> int:
    """Return global rank, or 0."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Return local rank on node, default from LOCAL_RANK env var or 0."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    """Return local world size on node, default from LOCAL_WORLD_SIZE env var or 1."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_node_rank() -> int:
    """Compute node rank from environment or from global and local ranks."""
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    return (get_global_rank() - get_local_rank()) // max(get_local_world_size(), 1)


def get_fs_local_rank() -> int:
    """Return filesystem-local rank, using FS_LOCAL_RANK env var or global/local rank."""
    shared_fs = os.environ.get("OLMO_SHARED_FS", "")
    if shared_fs:
        return int(os.environ.get("FS_LOCAL_RANK", str(get_global_rank())))
    return int(os.environ.get("FS_LOCAL_RANK", str(get_local_rank())))


def barrier() -> None:
    """Synchronize all processes in distributed group."""
    if is_distributed():
        dist.barrier()


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """Return peak GPU memory (MB) across all ranks; reset if requested."""
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    peak = torch.cuda.max_memory_allocated(device) / 1e6
    if is_distributed():
        tensor = torch.tensor([peak], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        peak = tensor.item()
    if reset:
        torch.cuda.reset_max_memory_allocated(device)
    return peak


def move_to_device(obj: Any, device: Optional[torch.device] = None) -> Any:
    """Recursively move tensors in obj to device."""
    if device is None:
        device = get_default_device()
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [move_to_device(x, device) for x in obj]
        return type(obj)(converted)
    return obj


def ensure_finite_(tensor: torch.Tensor, replace_neg_inf: bool = True, replace_pos_inf: bool = True) -> None:
    """In-place replace -inf/inf with finite min/max of dtype."""
    info = torch.finfo(tensor.dtype)
    if replace_neg_inf:
        tensor.masked_fill_(tensor == float("-inf"), info.min)
    if replace_pos_inf:
        tensor.masked_fill_(tensor == float("inf"), info.max)


def synchronize_value(value: V, device: torch.device = None, op: str = "broadcast") -> V:
    """Synchronize scalar value across processes; supports 'broadcast', 'mean', 'sum'."""
    if not is_distributed():
        return value
    device = device or get_default_device()
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if op == "broadcast":
        dist.broadcast(tensor, src=0)
    elif op in ("mean", "sum", "max", "min"):
        if op == "mean" or op == "sum":
            reduce_op = dist.ReduceOp.SUM
        elif op == "max":
            reduce_op = dist.ReduceOp.MAX
        else:
            reduce_op = dist.ReduceOp.MIN
        dist.all_reduce(tensor, op=reduce_op)
        if op == "mean":
            tensor /= get_world_size()
    else:
        raise ValueError(f"Unsupported op '{op}' for synchronize_value")
    out_val = type(value)(tensor.item())
    return out_val


def synchronize_flag(flag: bool, device: torch.device = None) -> bool:
    """Synchronize boolean flag by broadcasting from rank 0."""
    return bool(synchronize_value(flag, device, op="broadcast"))


def gc_cuda() -> None:
    """Collect Python GC and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    """Return 1D tensor of cumulative lengths for non-zero doc lengths."""
    non_zero = doc_lens[doc_lens != 0]
    cum = torch.cumsum(non_zero.to(torch.int32), dim=0)
    return torch.cat((torch.zeros(1, dtype=torch.int32, device=doc_lens.device), cum))


class SingleAccelerator(torch.nn.Module):
    """Wrapper for single-accelerator training (no DDP)."""
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


__all__ = [
    "get_default_device", "seed_all", "is_distributed", "get_world_size", "get_global_rank",
    "get_local_rank", "get_local_world_size", "get_node_rank", "get_fs_local_rank",
    "barrier", "peak_gpu_memory", "move_to_device", "ensure_finite_", "synchronize_value",
    "synchronize_flag", "gc_cuda", "get_cumulative_document_lengths", "SingleAccelerator"
]
