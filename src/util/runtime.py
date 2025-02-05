from __future__ import annotations
import os
import time
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Callable, TypeVar, Any, Optional, Tuple

T = TypeVar("T")
V = TypeVar("V", bool, int, float)

def _torch_dist_available_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def is_distributed() -> bool:
    if _torch_dist_available_initialized():
        return True
    return bool(int(os.environ.get("DISTRIBUTED", 0)))

def get_global_rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK") or os.environ.get("GLOBAL_RANK") or 0)

def get_local_rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(os.environ.get("LOCAL_RANK") or 0)
    except Exception:
        pass
    return int(os.environ.get("LOCAL_RANK") or 0)

def get_world_size() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or 1)

def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)

def get_node_rank() -> int:
    try:
        return int(os.environ.get("NODE_RANK") or 0)
    except Exception:
        return 0

def get_fs_local_rank() -> int:
    if os.environ.get("ELEVATION_SHARED_FS"):
        return int(os.environ.get("FS_LOCAL_RANK") or get_global_rank())
    return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())

def seed_all_rngs(seed: int) -> None:
    if seed < 0 or seed > 2 ** 32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def transfer_to_device(obj: T, device: torch.device) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: transfer_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [transfer_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(transfer_to_device(x, device) for x in obj)
    return obj

def ensure_finite_inplace(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False) -> None:
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)

def get_default_compute_device() -> torch.device:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_initialized", lambda: True)():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and getattr(torch.backends.mps, "is_available", lambda: False)():
        return torch.device("mps")
    return torch.device("cpu")

def dist_barrier() -> None:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

def peak_gpu_memory_mb(reset: bool = False) -> float | None:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
    if dist.is_available() and dist.is_initialized():
        peak_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_tensor, 0, op=dist.ReduceOp.MAX)
        peak_mb = float(peak_tensor.item())
    if reset:
        try:
            torch.cuda.reset_max_memory_allocated(device)
        except Exception:
            pass
    return peak_mb

def broadcast_value(value: V, device: torch.device) -> V:
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor(value, device=device)
        dist.broadcast(tensor, src=0)
        return tensor.item()
    return value

def broadcast_flag(flag: bool, device: torch.device) -> bool:
    return bool(broadcast_value(flag, device))

def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        torch.tensor([0], dtype=torch.int32, device=doc_lens.device),
        torch.cumsum(doc_lens.masked_select(doc_lens != 0), 0, dtype=torch.int32),
    ])

class SingleDeviceModule(torch.nn.Module):
    process_group = None
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def default_thread_count() -> int:
    return int(os.environ.get("NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))

def wait_for(cond: Callable[[], bool], desc: str, timeout: float = 10.0) -> None:
    start = time.monotonic()
    while not cond():
        time.sleep(0.5)
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"{desc} timed out")

def pass_through_fn(fn: Callable, *args, **kwargs):
    return fn(*args, **kwargs)

def find_local_checkpoint(folder: str, prefix: str = "model") -> Tuple[Optional[str], Optional[int]]:
    """Return the latest local checkpoint directory matching prefix and its step number."""
    if not os.path.isdir(folder):
        return None, None
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(folder):
        if name.startswith(prefix):
            try:
                step = int(''.join(filter(str.isdigit, name)))
                if step > best_step:
                    best_step = step
                    best_path = os.path.join(folder, name)
            except ValueError:
                continue
    if best_path is not None:
        return best_path, best_step
    return None, None

def get_progress_bar():
    from cached_path import get_download_progress
    return get_download_progress()

from queue import Queue
from threading import Thread
from itertools import cycle, islice
from src.exceptions import OLMoThreadError

def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)
    sentinel = object()
    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)
    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()
    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise OLMoThreadError(f"generator thread {thread_name} failed") from x
        else:
            yield x

def roundrobin(*iterables):
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
