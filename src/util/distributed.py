import os
import random
from typing import Any, TypeVar, Union, Optional

T = TypeVar("T")
V = TypeVar("V", bool, int, float)


def _torch_dist_available_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def is_distributed() -> bool:
    if _torch_dist_available_initialized():
        import torch.distributed as dist
        return dist.is_initialized()
    return bool(int(os.environ.get("DISTRIBUTED", 0)))


def get_global_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK") or os.environ.get("GLOBAL_RANK") or 0)


def get_local_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(os.environ.get("LOCAL_RANK") or 0)
    except Exception:
        pass
    return int(os.environ.get("LOCAL_RANK") or 0)


def get_world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_node_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(os.environ.get("NODE_RANK") or 0)
    except Exception:
        pass
    return int(os.environ.get("NODE_RANK") or 0)


def get_fs_local_rank() -> int:
    if os.environ.get("ELEVATION_SHARED_FS"):
        return get_global_rank()
    return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())


def seed_all_rngs(seed: int) -> None:
    if seed < 0 or seed > 2 ** 32 - 1:
        raise ValueError("Seed must be in [0, 2**32 - 1]")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

seed_all = seed_all_rngs


def transfer_to_device(obj: T, device: Any) -> T:
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: transfer_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [transfer_to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(transfer_to_device(v, device) for v in obj)
    except ImportError:
        pass
    return obj


def ensure_finite_inplace(x: Any, check_neg_inf: bool = True, check_pos_inf: bool = False) -> None:
    try:
        import torch
        if check_neg_inf:
            x[x == float('-inf')] = torch.finfo(x.dtype).min
        if check_pos_inf:
            x[x == float('inf')] = torch.finfo(x.dtype).max
    except ImportError:
        pass


def get_default_compute_device() -> Any:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


def dist_barrier() -> None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def peak_gpu_memory_mb(reset: bool = False) -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda")
        peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        return peak_mb
    except Exception:
        return None


def broadcast_value(value: V, device: Any) -> V:
    try:
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor([value], device=device)
            dist.broadcast(tensor, src=0)
            return tensor.item()
    except Exception:
        pass
    return value


def broadcast_flag(flag: bool, device: Any) -> bool:
    return bool(broadcast_value(flag, device))
