from __future__ import annotations

import io
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import reduce
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, cast

# Optional torch imports guarded to keep this module import-safe without runtime deps.
try:  # pragma: no cover - import-time guard
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - minimal fallback
    np = None  # type: ignore

try:  # pragma: no cover - import-time guard
    import torch  # type: ignore
    import torch.distributed.checkpoint as dist_cp  # type: ignore
    import torch.multiprocessing as mp  # type: ignore
    import torch.nn as nn  # type: ignore
    from packaging import version  # type: ignore
    from torch.distributed import _remote_device  # type: ignore
    from torch.distributed._shard._utils import narrow_tensor_by_index  # type: ignore
    from torch.distributed._shard.metadata import ShardMetadata  # type: ignore
    from torch.distributed._shard.sharded_tensor import ShardedTensor  # type: ignore
    from torch.distributed.checkpoint.filesystem import WriteResult, _StorageInfo  # type: ignore
    from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex  # type: ignore
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict  # type: ignore
    from torch.distributed.checkpoint.planner import LoadItemType, ReadItem  # type: ignore
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
    from torch.distributed.fsdp import StateDictType  # type: ignore
    from torch.distributed.fsdp.api import (  # type: ignore
        FullOptimStateDictConfig,
        FullStateDictConfig,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
    )
    from torch.futures import Future  # type: ignore
    from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore

    try:  # torch internal location changed between versions
        from torch.distributed.fsdp.flat_param import FlatParamHandle  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - version guard
        from torch.distributed.fsdp._flat_param import FlatParamHandle  # type: ignore
except Exception:  # pragma: no cover - import-time guard
    torch = None  # type: ignore
    dist_cp = None  # type: ignore
    mp = None  # type: ignore
    nn = None  # type: ignore
    version = None  # type: ignore
    _remote_device = None  # type: ignore
    ShardMetadata = None  # type: ignore
    ShardedTensor = None  # type: ignore
    WriteResult = None  # type: ignore
    _StorageInfo = None  # type: ignore
    Metadata = None  # type: ignore
    MetadataIndex = None  # type: ignore
    load_sharded_optimizer_state_dict = None  # type: ignore
    LoadItemType = None  # type: ignore
    ReadItem = None  # type: ignore
    FSDP = None  # type: ignore
    StateDictType = None  # type: ignore
    FullOptimStateDictConfig = None  # type: ignore
    FullStateDictConfig = None  # type: ignore
    ShardedOptimStateDictConfig = None  # type: ignore
    ShardedStateDictConfig = None  # type: ignore
    Future = None  # type: ignore
    DDP = None  # type: ignore
    FlatParamHandle = None  # type: ignore

from ..aliases import PathOrStr


# ===== Logging =====
from ..logger import get_logger

log = get_logger(__name__)


# ===== Exceptions =====
class CheckpointError(RuntimeError):
    """Checkpointing failure with a safe, serializable message."""


# ===== Utilities =====

def _atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(str(tmp_dir), str(final_dir))


def _cpu_threads_default() -> int:
    try:
        return max(1, (os.cpu_count() or 4))
    except Exception:
        return 4


def barrier() -> None:
    if torch is None:
        return
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def get_global_rank() -> int:
    if torch is None:
        return 0
    try:
        import torch.distributed as dist  # type: ignore

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def get_world_size() -> int:
    if torch is None:
        return 1
    try:
        import torch.distributed as dist  # type: ignore

        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        return 1


def get_local_rank() -> int:
    # Best-effort; many launchers export LOCAL_RANK.
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def get_local_world_size() -> int:
    try:
        return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    except Exception:
        return 1


def get_fs_local_rank() -> int:
    # Filesystem-local rank 0 per node; assume LOCAL_RANK works.
    return get_local_rank()


def wait_for(predicate, desc: str, timeout: float = 10.0, interval: float = 0.05) -> None:
    start = time.time()
    while True:
        try:
            if predicate():
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(desc)
        time.sleep(interval)


def _resource_path(base: PathOrStr, fname: str, *, local_cache: Optional[PathOrStr] = None) -> Path:
    # Local-only implementation; extend via a storage plugin as needed.
    base_path = Path(str(base).rstrip("/"))
    candidate = base_path / fname
    if candidate.is_file():
        return candidate
    if local_cache is not None:
        cache_candidate = Path(str(local_cache)) / fname
        if cache_candidate.is_file():
            return cache_candidate
    raise FileNotFoundError(candidate)


def _upload(source: Path, target_uri: str, *, save_overwrite: bool = False) -> None:
    # Local-only implementation: support "file://" or absolute/relative paths.
    if target_uri.startswith("file://"):
        target_path = Path(target_uri[len("file://") :])
    else:
        # Treat as filesystem path.
        target_path = Path(target_uri)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not save_overwrite:
        raise FileExistsError(target_path)
    shutil.copy2(source, target_path)


# ===== Optional safetensors support (read-only) =====
try:  # pragma: no cover
    from safetensors.torch import load_file as _safetensors_load_file  # type: ignore
except Exception:  # pragma: no cover
    _safetensors_load_file = None  # type: ignore


# ===== Public helpers =====
MODEL_AND_OPTIM_FOLDER = "model_and_optim"


def save_state_dict(
    checkpoint_dir: PathOrStr,
    fname: str,
    state_dict: Dict[str, Any],
    *,
    upload_to: Optional[str] = None,
    save_overwrite: bool = False,
    synchronize: bool = True,
) -> None:
    if torch is None:
        raise CheckpointError("torch is required for saving state dicts")
    checkpoint_dir = Path(checkpoint_dir)
    target_path = checkpoint_dir / fname
    if save_overwrite:
        target_path.unlink(missing_ok=True)
    elif target_path.is_file():
        raise FileExistsError(target_path)
    if synchronize:
        barrier()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if synchronize:
        barrier()
    torch.save(state_dict, target_path)
    if upload_to is not None:
        target = f"{upload_to.rstrip('/')}/{fname}"
        log.info("Uploading %s to %s...", target_path, target)
        _upload(target_path, target, save_overwrite=save_overwrite)


def load_state_dict(
    checkpoint_dir: PathOrStr,
    fname: str,
    *,
    local_cache: Optional[PathOrStr] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise CheckpointError("torch is required for loading state dicts")
    if fname.endswith(".pt") and _safetensors_load_file is not None:
        try:
            path = _resource_path(checkpoint_dir, fname[:-2] + "safetensors", local_cache=local_cache)
            return _safetensors_load_file(str(path))  # type: ignore
        except FileNotFoundError:
            pass
    path = _resource_path(checkpoint_dir, fname, local_cache=local_cache)
    return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore


def load_model_state(checkpoint_dir: PathOrStr, model: Any) -> None:
    if torch is None or dist_cp is None:
        raise CheckpointError("torch distributed checkpoint is required")
    state_dict_obj = {"model": model.state_dict()}
    reader = RemoteFileSystemReader(f"{str(checkpoint_dir).rstrip('/')}/{MODEL_AND_OPTIM_FOLDER}")
    dist_cp.load_state_dict(state_dict_obj, reader, no_dist=True)
    model.load_state_dict(state_dict_obj["model"])  # type: ignore


# ===== Remote I/O (local-only baseline; extend via plugin) =====
class RemoteFileSystemWriter:
    """Minimal local writer API compatible with dist_cp expectations.

    Extend or replace to support cloud storage. Local impl writes to filesystem and optionally mirrors to another path.
    """

    def __init__(
        self,
        path: PathOrStr,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: Optional[int] = None,
        per_thread_copy_ahead: int = 10_000_000,
        upload_to: Optional[str] = None,
        save_overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = max(1, thread_count or 1)
        self.per_thread_copy_ahead = per_thread_copy_ahead
        self.upload_to = None if upload_to is None else upload_to.rstrip("/")
        self.save_overwrite = save_overwrite

    def write_data(self, plan, planner):  # type: ignore[override]
        if dist_cp is None:
            raise CheckpointError("torch distributed checkpoint is required for write_data")
        fut = dist_cp.FileSystemWriter(  # type: ignore
            self.path,
            single_file_per_rank=self.single_file_per_rank,
            sync_files=self.sync_files,
            thread_count=self.thread_count,
            per_thread_copy_ahead=self.per_thread_copy_ahead,
        ).write_data(plan, planner)

        if self.upload_to is not None:
            files_to_upload: Set[str] = set()
            for write_result in fut.wait():  # type: ignore[attr-defined]
                files_to_upload.add(write_result.storage_data.relative_path)  # type: ignore[attr-defined]
            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                futures = []
                for fname in files_to_upload:
                    src = self.path / fname
                    dst = f"{self.upload_to}/{fname}"
                    log.info("Uploading %s to %s...", src, dst)
                    futures.append(executor.submit(_upload, src, dst, save_overwrite=self.save_overwrite))
                for ff in as_completed(futures):
                    ff.result()
        return fut

    def finish(self, metadata, results) -> None:  # type: ignore[override]
        if dist_cp is None:
            raise CheckpointError("torch distributed checkpoint is required for finish")
        dist_cp.FileSystemWriter(self.path).finish(metadata, results)  # type: ignore
        if self.upload_to is not None:
            src = self.path / ".metadata"
            dst = f"{self.upload_to}/.metadata"
            log.info("Uploading %s to %s...", src, dst)
            _upload(src, dst, save_overwrite=self.save_overwrite)


class RemoteFileSystemReader:
    """Local/remote reader abstraction.

    Local-only baseline; override _get_bytes to support cloud backends.
    """

    def __init__(self, path: PathOrStr, *, local_cache: Optional[PathOrStr] = None, thread_count: Optional[int] = None):
        if dist_cp is None:
            raise CheckpointError("torch distributed checkpoint is required for reader")
        self.path = str(path).rstrip("/")
        self.cache = None if local_cache is None else Path(local_cache)
        self.thread_count = max(1, thread_count or _cpu_threads_default())
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()  # type: ignore
        self._metadata: Optional[Metadata] = None  # type: ignore

    def _get_bytes(self, relative_path: str, offset: int, length: int) -> bytes:
        if self.cache is not None and (p := self.cache / relative_path).is_file():
            with open(p, "rb") as f:
                f.seek(offset)
                return f.read(length)
        else:
            with open(os.path.join(self.path, relative_path), "rb") as f:
                f.seek(offset)
                return f.read(length)

    def _get_content_for_read(self, read_item: ReadItem):  # type: ignore
        sinfo = self.storage_data[read_item.storage_index]  # type: ignore[index]
        content = self._get_bytes(sinfo.relative_path, sinfo.offset, sinfo.length)  # type: ignore[attr-defined]
        return (read_item, content)

    def read_data(self, plan, planner):  # type: ignore[override]
        if Future is None:
            raise CheckpointError("torch futures are required for reader.read_data")
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = [executor.submit(self._get_content_for_read, item) for item in plan.items]  # type: ignore[attr-defined]
            results = [f.result() for f in as_completed(futures)]

        for read_item, content in results:
            bytes_io = io.BytesIO(content)
            bytes_io.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:  # type: ignore[attr-defined]
                planner.load_bytes(read_item, bytes_io)
            else:
                tensor = cast("torch.Tensor", torch.load(bytes_io, map_location="cpu"))  # type: ignore
                tensor = narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)  # type: ignore
                target_tensor = planner.resolve_tensor(read_item).detach()
                assert target_tensor.size() == tensor.size(), (
                    f"req {read_item.storage_index} mismatch {target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()  # type: ignore
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:  # type: ignore[override]
        if self._metadata is None:
            with _resource_path(self.path, ".metadata", local_cache=self.cache).open("rb") as f:
                self._metadata = pickle.load(f)
        return self._metadata  # type: ignore[return-value]

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:  # type: ignore[override]
        del is_coordinator
        self.storage_data = metadata.storage_data  # type: ignore
        assert self.storage_data is not None

    def prepare_local_plan(self, plan):  # type: ignore[override]
        return plan

    def prepare_global_plan(self, global_plan):  # type: ignore[override]
        return global_plan


# ===== Checkpointer config and base =====
@dataclass
class CheckpointConfig:
    save_overwrite: bool = False
    sharded_strategy: Optional[str] = None  # "torch_new" | "torch_legacy" | "local"
    thread_count: int = field(default_factory=_cpu_threads_default)


class CheckpointerBase(metaclass=ABCMeta):
    def __init__(self, cfg: CheckpointConfig, thread_count: Optional[int] = None):
        self.cfg = cfg
        self.thread_count = thread_count or cfg.thread_count or _cpu_threads_default()

    @abstractmethod
    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: Any,
        optim: Any,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: Any,
        optim: Any,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        raise NotImplementedError

    @contextmanager
    def _staging_dir(self, dir: PathOrStr) -> Generator[Path, None, None]:
        checkpoint_dir = Path(dir)
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            if self.cfg.save_overwrite:
                if get_fs_local_rank() == 0:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
            else:
                raise FileExistsError(checkpoint_dir)
        barrier()
        tmp = checkpoint_dir.with_name(checkpoint_dir.name + "-tmp")
        if get_fs_local_rank() == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
        wait_for(lambda: tmp.exists(), "Waiting for checkpoint staging directory", timeout=10.0)
        barrier()
        yield tmp
        barrier()
        if get_fs_local_rank() == 0:
            try:
                _atomic_replace_dir(tmp, checkpoint_dir)
            except FileNotFoundError:
                if not checkpoint_dir.exists():
                    raise
        wait_for(lambda: checkpoint_dir.exists(), "Waiting for final checkpoint directory", timeout=10.0)
        barrier()

    def _save_config(self, dir: PathOrStr, *, upload_to: Optional[str] = None) -> None:
        cfg_path = Path(dir) / "config.json"
        if get_global_rank() == 0:
            try:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "save_overwrite": self.cfg.save_overwrite,
                        "sharded_strategy": self.cfg.sharded_strategy,
                        "thread_count": self.thread_count,
                    }, f)
                if upload_to is not None:
                    _upload(cfg_path, f"{upload_to}/config.json", save_overwrite=self.cfg.save_overwrite)
            except Exception as e:
                raise CheckpointError(str(e))


# ===== Full (single-file) checkpointer =====
class FullCheckpointer(CheckpointerBase):
    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: Any,
        optim: Any,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise CheckpointError("torch is required")
        with self._staging_dir(dir) as checkpoint_dir:
            if FSDP is not None and isinstance(dist_model, FSDP):  # type: ignore
                with FSDP.state_dict_type(  # type: ignore[attr-defined]
                    dist_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,  # type: ignore
                    state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),  # type: ignore
                    optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),  # type: ignore
                ):
                    model_state = dist_model.state_dict()  # type: ignore[attr-defined]
                    self._write_model_dict(model_state, checkpoint_dir, upload_to)
                    optim_state = FSDP.optim_state_dict(dist_model, optim)  # type: ignore[attr-defined]
                    self._write_optim_dict(optim_state, checkpoint_dir, upload_to)
            elif DDP is not None and isinstance(dist_model, (DDP,)):
                model_state = dist_model.module.state_dict()  # type: ignore[attr-defined]
                self._write_model_dict(model_state, checkpoint_dir, upload_to)
                self._write_optim_dict(optim.state_dict(), checkpoint_dir, upload_to)
            else:
                model_state = dist_model.state_dict()  # type: ignore[attr-defined]
                self._write_model_dict(model_state, checkpoint_dir, upload_to)
                self._write_optim_dict(getattr(optim, "state_dict", lambda: {})(), checkpoint_dir, upload_to)

            if get_global_rank() == 0:
                save_state_dict(checkpoint_dir, "train.pt", trainer_state, upload_to=upload_to, save_overwrite=self.cfg.save_overwrite, synchronize=False)
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: Any,
        optim: Any,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        if torch is None:
            raise CheckpointError("torch is required")
        if FSDP is not None and isinstance(dist_model, FSDP):  # type: ignore
            with FSDP.state_dict_type(  # type: ignore[attr-defined]
                dist_model,
                state_dict_type=StateDictType.FULL_STATE_DICT,  # type: ignore
                state_dict_config=FullStateDictConfig(rank0_only=False, offload_to_cpu=True),  # type: ignore
                optim_state_dict_config=FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),  # type: ignore
            ):
                with torch.no_grad():  # type: ignore
                    state_to_load = load_state_dict(load_path, "model.pt", local_cache=local_cache, map_location="cpu")
                    dist_model.load_state_dict(state_to_load)  # type: ignore[attr-defined]
                if load_optimizer_state:
                    optim_state = load_state_dict(load_path, "optim.pt", local_cache=local_cache, map_location="cpu")
                    _load_fsdp_optim_state(dist_model, optim, optim_state)  # type: ignore
        elif DDP is not None and isinstance(dist_model, (DDP,)):
            state_to_load = load_state_dict(load_path, "model.pt", local_cache=local_cache, map_location="cpu")
            dist_model.module.load_state_dict(state_to_load, strict=True)  # type: ignore[attr-defined]
            if load_optimizer_state:
                optim.load_state_dict(load_state_dict(load_path, "optim.pt", local_cache=local_cache, map_location="cpu"))
            barrier()
        else:
            state_to_load = load_state_dict(load_path, "model.pt", local_cache=local_cache, map_location="cpu")
            dist_model.load_state_dict(state_to_load)  # type: ignore[attr-defined]
            if load_optimizer_state and hasattr(optim, "load_state_dict"):
                optim.load_state_dict(load_state_dict(load_path, "optim.pt", local_cache=local_cache, map_location="cpu"))

        try:
            trainer_state = load_state_dict(load_path, "train.pt", local_cache=local_cache)
        except FileNotFoundError:
            trainer_state = {}
        barrier()
        return trainer_state

    def _write_model_dict(self, model_state_dict, checkpoint_dir, upload_to):
        if get_global_rank() == 0:
            log.info("Saving model state...")
            save_state_dict(checkpoint_dir, "model.pt", model_state_dict, upload_to=upload_to, save_overwrite=self.cfg.save_overwrite, synchronize=False)
        barrier()

    def _write_optim_dict(self, optim_state_dict, checkpoint_dir, upload_to):
        if get_global_rank() == 0:
            log.info("Saving optim state...")
            save_state_dict(checkpoint_dir, "optim.pt", optim_state_dict, upload_to=upload_to, save_overwrite=self.cfg.save_overwrite, synchronize=False)
        barrier()


# ===== FSDP helpers =====
def _load_fsdp_optim_state(fsdp_model: Any, optim: Any, optim_state: Dict[str, Any]) -> None:
    if torch is None or FSDP is None:
        raise CheckpointError("FSDP required to load optimizer state")
    if version is not None and version.parse(torch.__version__) < version.parse("2.1.0"):  # type: ignore
        flattened_osd = FSDP.optim_state_dict_to_load(optim_state, fsdp_model, optim)  # type: ignore
    else:
        flattened_osd = FSDP.optim_state_dict_to_load(fsdp_model, optim, optim_state)  # type: ignore
    for state in flattened_osd.get("state", {}).values():
        for k in list(state.keys()):
            try:
                state[k] = state[k].cpu()
            except Exception:
                pass
    optim.load_state_dict(flattened_osd)


# ===== New-style sharded (torch.distributed.checkpoint) =====
class TorchNewStyleShardedCheckpointer(CheckpointerBase):
    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: Any,
        optim: Any,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        if FSDP is None or dist_cp is None:
            raise CheckpointError("torch distributed checkpoint with FSDP required")
        if not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("TorchNewStyleShardedCheckpointer requires an FSDP-wrapped model")
        with self._staging_dir(dir) as checkpoint_dir:
            target_dir = Path(checkpoint_dir)
            with FSDP.state_dict_type(  # type: ignore[attr-defined]
                dist_model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,  # type: ignore
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),  # type: ignore
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),  # type: ignore
            ):
                payload = {"model": dist_model.state_dict(), "optim": FSDP.optim_state_dict(dist_model, optim)}  # type: ignore[attr-defined]
                dist_cp.save_state_dict(
                    payload,
                    RemoteFileSystemWriter(
                        target_dir / MODEL_AND_OPTIM_FOLDER,
                        upload_to=None if upload_to is None else f"{upload_to.rstrip('/')}/{MODEL_AND_OPTIM_FOLDER}",
                        save_overwrite=self.cfg.save_overwrite,
                    ),
                )
            save_state_dict(
                target_dir,
                f"train/rank{get_global_rank()}.pt",
                trainer_state,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )
            self._save_config(target_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: Any,
        optim: Any,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        if FSDP is None or dist_cp is None:
            raise CheckpointError("torch distributed checkpoint with FSDP required")
        if not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("TorchNewStyleShardedCheckpointer requires an FSDP-wrapped model")
        load_root = str(load_path).rstrip("/")
        with FSDP.state_dict_type(  # type: ignore[attr-defined]
            dist_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,  # type: ignore
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),  # type: ignore
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),  # type: ignore
        ):
            model_state = {"model": dist_model.state_dict()}  # type: ignore[attr-defined]
            reader = RemoteFileSystemReader(
                f"{load_root}/{MODEL_AND_OPTIM_FOLDER}",
                local_cache=None if local_cache is None else Path(local_cache) / MODEL_AND_OPTIM_FOLDER,
            )
            dist_cp.load_state_dict(model_state, reader)  # type: ignore
            dist_model.load_state_dict(model_state["model"])  # type: ignore[attr-defined]
            if load_optimizer_state:
                optim_state = load_sharded_optimizer_state_dict(  # type: ignore
                    model_state_dict=model_state["model"],
                    optimizer_key="optim",
                    storage_reader=reader,
                )
                for s in optim_state["optim"]["state"].values():  # type: ignore
                    for k in list(s.keys()):
                        try:
                            s[k] = s[k].cpu()
                        except Exception:
                            pass
                _load_fsdp_optim_state(dist_model, optim, optim_state["optim"])  # type: ignore
        try:
            trainer_state = load_state_dict(load_path, f"train/rank{get_global_rank()}.pt", local_cache=local_cache)
        except FileNotFoundError:
            trainer_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)
        barrier()
        return trainer_state


# ===== Legacy sharded (rank shards via torch.save) =====
class TorchLegacyShardedCheckpointer(CheckpointerBase):
    def __init__(self, cfg: CheckpointConfig, thread_count: Optional[int] = None, use_shared_mem_impl: bool = False):
        super().__init__(cfg, thread_count)
        self.use_shared_mem_impl = use_shared_mem_impl

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: Any,
        optim: Any,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        if FSDP is None:
            raise CheckpointError("FSDP required for legacy sharded checkpointing")
        if not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("TorchLegacyShardedCheckpointer requires an FSDP-wrapped model")
        with self._staging_dir(dir) as checkpoint_dir:
            with FSDP.state_dict_type(  # type: ignore[attr-defined]
                dist_model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,  # type: ignore
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),  # type: ignore
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),  # type: ignore
            ):
                state_dict = {
                    "model": dist_model.state_dict(),  # type: ignore[attr-defined]
                    "optim": FSDP.optim_state_dict(dist_model, optim),  # type: ignore[attr-defined]
                    **trainer_state,
                }
                save_state_dict(
                    checkpoint_dir,
                    f"rank{get_global_rank()}.pt",
                    state_dict,
                    upload_to=upload_to,
                    save_overwrite=self.cfg.save_overwrite,
                )
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: Any,
        optim: Any,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        if FSDP is None:
            raise CheckpointError("FSDP required for legacy sharded checkpointing")
        if not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("TorchLegacyShardedCheckpointer requires an FSDP-wrapped model")
        with FSDP.state_dict_type(  # type: ignore[attr-defined]
            dist_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,  # type: ignore
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),  # type: ignore
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),  # type: ignore
        ):
            state_dict = load_state_dict(load_path, f"rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu")
            dist_model.load_state_dict(state_dict["model"])  # type: ignore[attr-defined]
            del state_dict["model"]
            if load_optimizer_state:
                _load_fsdp_optim_state(dist_model, optim, state_dict["optim"])  # type: ignore
            del state_dict["optim"]
        barrier()
        return state_dict

    # Unshard helpers and implementations adapted and simplified.
    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if local_cache is not None:
            raise CheckpointError("unshard_checkpoint only supports local files for legacy sharded checkpoints")
        if torch is None:
            raise CheckpointError("torch is required")
        full_state = self._unshard(load_path, device or torch.device("cpu"), skip_keys={"rng"})  # type: ignore
        model_state = full_state.pop("model")
        optim_state = full_state.pop("optim", None)
        return model_state, (optim_state if load_optimizer_state else None), (full_state if load_trainer_state else None)

    def _unshard(self, input_dir: PathOrStr, device: Any, skip_keys: Optional[Set[str]] = None):
        input_dir = Path(input_dir)
        skip = skip_keys or set()
        shards_dict = {}
        with ThreadPoolExecutor() as executor:
            for shard_name in input_dir.glob("rank*.pt"):
                rank = int(shard_name.name[4:-3])
                shards_dict[rank] = executor.submit(torch.load, shard_name, map_location="cpu")  # type: ignore
        shards: List[Dict[str, Any]] = [None] * len(shards_dict)  # type: ignore
        for rank, fut in shards_dict.items():
            shard = fut.result()
            for k in list(shard.keys()):
                if k in skip:
                    del shard[k]
            shards[rank] = shard
        return self._unshard_object(shards, device=device)

    def _unshard_object(self, os_list: List[Any], device: Any) -> Any:
        a0 = os_list[0]
        assert all(type(o) is type(a0) for o in os_list)
        if isinstance(a0, str):
            assert all(o == a0 for o in os_list)
            return a0
        elif isinstance(a0, (list, tuple, set)):
            assert all(len(o) == len(a0) for o in os_list)
            return a0.__class__(self._unshard_object(o, device=device) for o in zip(*os_list))
        elif isinstance(a0, dict):
            assert all(o.keys() == a0.keys() for o in os_list)
            return {k: self._unshard_object([o[k] for o in os_list], device=device) for k in a0.keys()}
        elif ShardedTensor is not None and isinstance(a0, ShardedTensor):  # type: ignore
            return self._gather(os_list, device=device)
        else:
            return a0

    def _gather(self, shards: List[Any], device: Any) -> Any:
        shard0_md = shards[0].metadata()
        def shard_size(md):
            return reduce((lambda x, y: x * y), md.shard_sizes)  # type: ignore[attr-defined]
        rank_sizes = defaultdict(int)
        placement = {}
        for md in shard0_md.shards_metadata:
            rank = cast(_remote_device, md.placement).rank()  # type: ignore
            placement[md] = (rank, rank_sizes[rank])
            rank_sizes[rank] += shard_size(md)
        max_rank_size = max(rank_sizes.values()) if rank_sizes else 0
        gather_list: List[Any] = [torch.empty((max_rank_size,)) for _ in range(len(rank_sizes))]  # type: ignore
        datas = []
        with torch.no_grad():  # type: ignore
            for shard in shards:
                data = torch.empty(max_rank_size)  # type: ignore
                for local_shard in shard.local_shards():
                    src = local_shard.tensor.flatten()
                    off = placement[local_shard.metadata][1]
                    data[off : off + src.numel()].copy_(src)
                datas.append(data)
        for r, data in enumerate(datas):
            gather_list[r].copy_(data)
        full_size = shard0_md.size
        out = torch.empty(*full_size, dtype=shard0_md.tensor_properties.dtype, device=device)  # type: ignore
        dims = len(full_size)
        for md in shard0_md.shards_metadata:
            rank, off = placement[md]
            tensor = gather_list[rank]
            tensor = tensor[off : off + shard_size(md)]
            tensor = tensor.view(md.shard_sizes)
            view = out
            for d in range(dims):
                view = view.narrow(d, md.shard_offsets[d], md.shard_sizes[d])
                
            view.copy_(tensor)
        return out


# ===== Local-sharded (flat param) checkpointer =====
@dataclass
class _LocalShardedMeta:
    world_size: int = field(default_factory=get_world_size)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"world_size": self.world_size}))

    @classmethod
    def load(cls, path: Path) -> "_LocalShardedMeta":
        data = json.loads(path.read_text())
        m = cls()
        m.world_size = int(data.get("world_size", 1))
        return m


@dataclass
class _FlatParamShard:
    full_shape: Any
    shard_offsets: Tuple[int, int]
    shard_data: Optional[Any]

    def copy_into(self, full_tensor: Any) -> None:
        assert self.shard_data is not None
        view = full_tensor.view(-1)[self.shard_offsets[0] : self.shard_offsets[1] + 1]
        assert self.shard_data.shape == view.shape
        view.copy_(self.shard_data)


class LocalShardedCheckpointer(CheckpointerBase):
    _META_KEYS = (
        "_fqns",
        "_shard_param_offsets",
        "_shard_indices",
        "_numels",
        "_numels_with_padding",
        "_shapes",
        "_shard_numel_padded",
        "_shard_param_infos",
    )

    def _fsdp_modules(self, fsdp_model: Any) -> List[Tuple[str, Any]]:
        mods = []
        for n, m in fsdp_model.named_modules():  # type: ignore[attr-defined]
            if isinstance(m, FSDP):  # type: ignore
                mods.append((n, m))
        return mods

    def _prepare(self, fsdp_model: Any) -> None:
        if torch is None:
            return
        try:
            from torch.distributed.fsdp._runtime_utils import _lazy_init  # type: ignore
            if torch.cuda.is_available():  # type: ignore
                torch.cuda.synchronize()  # type: ignore
            _lazy_init(fsdp_model, fsdp_model)  # type: ignore
        except Exception:
            pass

    def _handles(self, fsdp_model: Any) -> List[Any]:
        if version is not None and version.parse(torch.__version__) < version.parse("2.1.0"):  # type: ignore
            return fsdp_model._handles  # type: ignore[attr-defined]
        elif version is not None and version.parse(torch.__version__) < version.parse("2.3.0"):  # type: ignore
            if hasattr(fsdp_model, "_handle") and fsdp_model._handle is not None:  # type: ignore[attr-defined]
                return [fsdp_model._handle]  # type: ignore
            return []
        else:
            raise NotImplementedError("FSDP internals changed; update _handles() for newer torch")

    @torch.no_grad()  # type: ignore
    def _flat_param_state(self, fsdp_model: Any) -> Dict[str, Any]:
        self._prepare(fsdp_model)
        modules = []
        for module_fqn, mod in self._fsdp_modules(fsdp_model):
            hds = []
            for h in self._handles(mod):
                data: Dict[str, Any] = {}
                flat_param = h.flat_param  # type: ignore[attr-defined]
                data["flat_param.data"] = flat_param.detach()
                for k in self._META_KEYS:
                    if hasattr(flat_param, k):
                        data[f"flat_param.{k}"] = getattr(flat_param, k)
                hds.append(data)
            modules.append({"handles": hds, "name": module_fqn})
        return {"modules": modules}

    @torch.no_grad()  # type: ignore
    def _load_flat_state(self, fsdp_model: Any, model_state: Dict[str, Any]) -> None:
        self._prepare(fsdp_model)
        mods = self._fsdp_modules(fsdp_model)
        assert len(model_state["modules"]) == len(mods)
        for (_, mod), mdata in zip(mods, model_state["modules"]):
            handles = self._handles(mod)
            assert len(handles) == len(mdata["handles"])  # type: ignore[index]
            for h, d in zip(handles, mdata["handles"]):  # type: ignore[index]
                flat_param = h.flat_param  # type: ignore[attr-defined]
                for k in self._META_KEYS:
                    if hasattr(flat_param, k):
                        assert getattr(flat_param, k) == d[f"flat_param.{k}"]
                flat_param.copy_(d["flat_param.data"])  # type: ignore[index]

    def _save_meta(self, dir: PathOrStr, *, upload_to: Optional[str] = None) -> None:
        if get_fs_local_rank() == 0:
            log.info("Saving metadata...")
            meta = _LocalShardedMeta()
            meta_path = Path(dir) / "metadata.json"
            meta.save(meta_path)
            if upload_to is not None and get_global_rank() == 0:
                _upload(meta_path, f"{upload_to}/metadata.json", save_overwrite=self.cfg.save_overwrite)

    def _load_meta(self, load_path: PathOrStr, *, local_cache: Optional[PathOrStr] = None) -> _LocalShardedMeta:
        return _LocalShardedMeta.load(_resource_path(load_path, "metadata.json", local_cache=local_cache))

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: Any,
        optim: Any,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        if FSDP is None or not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("LocalShardedCheckpointer requires an FSDP-wrapped model")
        with self._staging_dir(dir) as checkpoint_dir:
            log.info("Saving local FSDP flat params data...")
            save_state_dict(
                checkpoint_dir,
                f"model/rank{get_global_rank()}.pt",
                self._flat_param_state(dist_model),
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )
            log.info("Saving local optimizer state...")
            save_state_dict(
                checkpoint_dir,
                f"optim/rank{get_global_rank()}.pt",
                optim.state_dict(),
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )
            log.info("Saving trainer state...")
            save_state_dict(
                checkpoint_dir,
                f"train/rank{get_global_rank()}.pt",
                trainer_state,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )
            self._save_meta(checkpoint_dir, upload_to=upload_to)
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: Any,
        optim: Any,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        meta = self._load_meta(load_path, local_cache=local_cache)
        assert meta.world_size == get_world_size()
        if FSDP is None or not isinstance(dist_model, FSDP):  # type: ignore
            raise CheckpointError("LocalShardedCheckpointer requires an FSDP-wrapped model")
        model_state = load_state_dict(load_path, f"model/rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu")
        self._load_flat_state(dist_model, model_state)
        if load_optimizer_state:
            optim_state = load_state_dict(load_path, f"optim/rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu")
            # Strip non-essential statistics that may cause mismatches on load.
            for pid in list(optim_state.get("state", {}).keys()):
                state = optim_state["state"][pid]
                if "grad_norm_exp_avg" in state:
                    del state["grad_norm_exp_avg"]
                if len(state) == 0:
                    del optim_state["state"][pid]
            optim.load_state_dict(optim_state)
        trainer_state = load_state_dict(load_path, f"train/rank{get_global_rank()}.pt", local_cache=local_cache)
        barrier()
        return trainer_state

    def _iter_flat_param_shards(self, model_state: Dict[str, Any]) -> Generator[Tuple[str, _FlatParamShard], None, None]:
        for module_data in model_state["modules"]:
            module_prefix = module_data["name"].replace("_fsdp_wrapped_module.", "")
            for handle in module_data["handles"]:
                flat_data = handle["flat_param.data"]
                if (handle.get("flat_param._shard_numel_padded", 0)) > 0:
                    assert (flat_data[-handle["flat_param._shard_numel_padded"] :] == 0).all()
                if "flat_param._shard_indices" in handle:
                    param_start = handle["flat_param._shard_indices"][0]
                    cur = 0
                    for rel_fqn, full_shape, (off_s, off_e) in zip(
                        handle["flat_param._fqns"][param_start:],
                        handle["flat_param._shapes"][param_start:],
                        handle["flat_param._shard_param_offsets"],
                    ):
                        root_fqn = rel_fqn if not module_prefix else f"{module_prefix}.{rel_fqn}"
                        numel = off_e - off_s + 1
                        shard = _FlatParamShard(full_shape=full_shape, shard_offsets=(off_s, off_e), shard_data=flat_data[cur : cur + numel])
                        cur += numel
                        yield root_fqn, shard
                else:
                    for rel_fqn, full_shape, spi in zip(
                        handle["flat_param._fqns"],
                        handle["flat_param._shapes"],
                        handle["flat_param._shard_param_infos"],
                    ):
                        if not spi.in_shard:
                            continue
                        root_fqn = rel_fqn if not module_prefix else f"{module_prefix}.{rel_fqn}"
                        shard = _FlatParamShard(
                            full_shape=full_shape,
                            shard_offsets=(spi.intra_param_start_idx, spi.intra_param_end_idx),
                            shard_data=flat_data[spi.offset_in_shard : spi.offset_in_shard + spi.numel_in_shard],
                        )
                        yield root_fqn, shard

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if torch is None:
            raise CheckpointError("torch is required")
        device = device or torch.device("cpu")  # type: ignore
        meta = self._load_meta(load_path, local_cache=local_cache)
        log.info("Gathering model shards (%d ranks)...", meta.world_size)
        model_paths = self._gather_paths(load_path, "model", meta.world_size, local_cache=local_cache)
        full_model: Dict[str, Any] = {}
        flat_meta: Dict[int, Dict[str, _FlatParamShard]] = defaultdict(dict)
        for rank, path in enumerate(model_paths):
            log.info("Loading model shard from rank %d...", rank)
            model_state = torch.load(path, map_location="cpu")  # type: ignore
            for root_fqn, shard in self._iter_flat_param_shards(model_state):
                if root_fqn not in full_model:
                    assert shard.shard_data is not None
                    full_model[root_fqn] = torch.empty(shard.full_shape, dtype=shard.shard_data.dtype, device=device)  # type: ignore
                    full_model[root_fqn].fill_(torch.nan)
                shard.copy_into(full_model[root_fqn])
                flat_meta[rank][root_fqn] = replace(shard, shard_data=None)
        for k, t in full_model.items():
            if torch.isnan(t).any():  # type: ignore
                raise CheckpointError(f"Parameter '{k}' contains NaNs after unshard")
        trainer_state: Optional[Dict[str, Any]] = None
        if load_trainer_state:
            trainer_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)
        if not load_optimizer_state:
            return full_model, None, trainer_state
        log.info("Gathering optimizer shards (%d ranks)...", meta.world_size)
        optim_paths = self._gather_paths(load_path, "optim", meta.world_size, local_cache=local_cache)
        full_optim: Dict[str, Any] = {"state": defaultdict(dict)}
        fqn_to_id: Dict[str, int] = {}
        id_to_fqn: Dict[int, str] = {}
        for rank, path in enumerate(optim_paths):
            log.info("Loading optim shard from rank %d...", rank)
            optim_state = torch.load(path, map_location="cpu")  # type: ignore
            if "param_groups" not in full_optim:
                full_optim["param_groups"] = optim_state["param_groups"]
            else:
                assert full_optim["param_groups"] == optim_state["param_groups"]
            if not fqn_to_id or not id_to_fqn:
                for grp in full_optim["param_groups"]:
                    for fqn, pid in zip(grp["param_names"], grp["params"]):
                        fqn = fqn.replace("_fsdp_wrapped_module.", "")
                        fqn_to_id[fqn] = pid
                        id_to_fqn[pid] = fqn
            for pid, shard_state in optim_state["state"].items():
                fqn = id_to_fqn[pid]
                meta_shard = flat_meta[rank].get(fqn)
                full_state = full_optim["state"][pid]
                for key, val in shard_state.items():
                    if isinstance(val, torch.Tensor) and val.shape != torch.Size([]):  # type: ignore
                        assert meta_shard is not None
                        if key not in full_state:
                            full_state[key] = torch.empty(meta_shard.full_shape, dtype=val.dtype, device=device)
                        replace(meta_shard, shard_data=val).copy_into(full_state[key])
                    else:
                        if key not in full_state:
                            full_state[key] = val
        for grp in full_optim.get("param_groups", []):
            grp["param_names"] = [n.replace("_fsdp_wrapped_module.", "") for n in grp["param_names"]]
        return full_model, full_optim, trainer_state

    def _gather_paths(
        self,
        load_path: PathOrStr,
        kind: str,
        world_size: int,
        *,
        local_cache: Optional[PathOrStr] = None,
    ) -> List[Path]:
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futs = [executor.submit(self._one_path, load_path, kind, r, local_cache) for r in range(world_size)]
            results: Dict[int, Path] = {}
            for fut in as_completed(futs):
                r, p = fut.result()
                results[r] = p
        return [results[r] for r in range(world_size)]

    def _one_path(self, load_path: PathOrStr, kind: str, rank: int, local_cache: Optional[PathOrStr]) -> Tuple[int, Path]:
        fname = f"{kind}/rank{rank}.pt"
        return rank, _resource_path(str(load_path).rstrip("/"), fname, local_cache=local_cache)


# ===== Builder =====
def build_sharded_checkpointer(cfg: CheckpointConfig, *, name: Optional[str] = None, use_shared_mem_impl: bool = False) -> CheckpointerBase:
    key = (name or cfg.sharded_strategy or "").lower()
    if key in ("torch_new", "new", "dist"):
        return TorchNewStyleShardedCheckpointer(cfg)
    if key in ("torch_legacy", "legacy"):
        return TorchLegacyShardedCheckpointer(cfg, use_shared_mem_impl=use_shared_mem_impl)
    if key in ("local", "flat"):
        return LocalShardedCheckpointer(cfg)
    return FullCheckpointer(cfg)


# ===== Backward-compatible simple local JSON checkpoint store =====
@dataclass
class LocalCheckpointStore:
    save_folder: PathOrStr

    def _dir(self, name: str) -> Path:
        return Path(str(self.save_folder)) / name

    def save(self, name: str = "model", state_dict: Optional[Dict[str, Any]] = None) -> str:
        base = Path(str(self.save_folder))
        base.mkdir(parents=True, exist_ok=True)
        final_dir = self._dir(name)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{name}.tmp.", dir=str(base)))
        with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
        if state_dict is not None:
            with open(tmp_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump(state_dict, f)
        _atomic_replace_dir(tmp_dir, final_dir)
        try:
            with open(base / "latest.txt", "w", encoding="utf-8") as f:
                f.write(name)
        except Exception:
            pass
        return str(final_dir)

    def load(self, name: str = "model") -> Optional[Dict[str, Any]]:
        folder = self._dir(name)
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            return None
        result: Dict[str, Any] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            result["meta"] = json.load(f)
        state_path = folder / "state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                result["state"] = json.load(f)
        return result

    def find_latest(self) -> Optional[str]:
        base = Path(str(self.save_folder))
        marker = base / "latest.txt"
        if marker.exists():
            try:
                return marker.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass
        try:
            entries = [
                (e.name, e.stat().st_mtime)
                for e in base.iterdir()
                if e.is_dir()
            ]
            if not entries:
                return None
            entries.sort(key=lambda x: x[1], reverse=True)
            return entries[0][0]
        except Exception:
            return None

    def prune(self, keep: int, predicate: Optional[callable] = None) -> None:
        try:
            base = Path(str(self.save_folder))
            entries = [
                (e.name, e.stat().st_mtime)
                for e in base.iterdir()
                if e.is_dir() and (predicate(e.name) if predicate else True)
            ]
            entries.sort(key=lambda x: x[1], reverse=True)
            for name, _ in entries[keep:]:
                shutil.rmtree(base / name, ignore_errors=True)
        except Exception:
            return


__all__ = [
    # helpers
    "save_state_dict",
    "load_state_dict",
    "load_model_state",
    "RemoteFileSystemWriter",
    "RemoteFileSystemReader",
    # configs and base
    "CheckpointConfig",
    "CheckpointerBase",
    # strategies
    "FullCheckpointer",
    "TorchNewStyleShardedCheckpointer",
    "TorchLegacyShardedCheckpointer",
    "LocalShardedCheckpointer",
    "build_sharded_checkpointer",
    # simple store
    "LocalCheckpointStore",
]
