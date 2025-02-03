from __future__ import annotations

import enum
from dataclasses import asdict, is_dataclass
import os
from typing import Dict, Iterable, Tuple, Optional, Callable, Union, Any
import logging
from .logger import get_logger
import socket
import sys
import warnings
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
import gzip
import io
import json
import re
import time
from itertools import cycle, islice
from urllib.parse import urlparse
from functools import lru_cache as cache

import requests
import datasets
import boto3
import botocore.exceptions as boto_exceptions
from botocore.config import Config
from google.api_core.retry import Retry as GCSRetry, if_transient_error as gcs_is_transient_error
from cached_path.schemes import SchemeClient, add_scheme_client
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.progress import Progress
from rich.text import Text
from rich.traceback import Traceback

from .aliases import PathOrStr
from .compatability.data import get_data_path
from .exceptions import (
    CliError,
    EnvironmentError,
    NetworkError,
    ThreadError,
)


# Update enum name to be clearer
class StringEnum(str, enum.Enum):
    """String-based Enum for compatibility with older Python versions."""
    def __str__(self) -> str:
        return self.value


def dataclass_asdict_shallow(obj: Any) -> Dict[str, Any]:
    """Return a shallow asdict for dataclasses, else wrap into a dict."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}

def flatten_dict(
    dictionary: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
    include_lists: bool = False,
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary.

    Args:
        dictionary: The nested dictionary.
        parent_key: Prefix for keys in the flattened dict.
        separator: Separator between parent and child keys.
        include_lists: Whether to convert lists to dicts with integer keys.

    Returns:
        Flattened dictionary.
    """
    items: Dict[str, Any] = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, list) and include_lists:
            value = {str(i): v for i, v in enumerate(value)}
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, separator, include_lists))
        else:
            items[new_key] = value
    return items


# Rename checkpoint finder to find_local_checkpoint
def find_local_checkpoint(
    folder: str,
    prefix: str = "model"
) -> Tuple[Optional[str], Optional[int]]:
    """
    Return the latest local checkpoint directory matching prefix, with step number.
    """
    if not os.path.isdir(folder):
        return None, None
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(folder):
        if not name.startswith(prefix):
            continue
        try:
            step = int(name.split("-")[-1]) if "-" in name else 0
        except ValueError:
            step = 0
        path = os.path.join(folder, name)
        if os.path.isdir(path) and step >= best_step:
            best_step = step
            best_path = path
    return best_path, (best_step if best_step >= 0 else None)


# Deprecate old find_latest_checkpoint alias
find_latest_checkpoint = find_local_checkpoint


# Add remote URL checker
def is_remote_url(path: PathOrStr) -> bool:
    """Return True if path is a remote URL (http, https, s3, gs, weka, r2)."""
    return re.match(r"^[a-z0-9]+://", str(path)) is not None


# dynamic log context
_log_extra_fields: Dict[str, object] = {}
log = get_logger(__name__)

class LogFilterType(str, enum.Enum):
    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"
    all_ranks = "all_ranks"

# Logging setup and context

def log_extra_field(name: str, value: object) -> None:
    if value is None:
        _log_extra_fields.pop(name, None)
    else:
        _log_extra_fields[name] = value


def setup_logging(filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    log_extra_field("hostname", socket.gethostname())
    if is_distributed():
        log_extra_field("node_rank", get_node_rank())
        log_extra_field("local_rank", get_local_rank())
        log_extra_field("global_rank", get_global_rank())
    else:
        log_extra_field("node_rank", 0)
        log_extra_field("local_rank", 0)
        log_extra_field("global_rank", 0)

    orig_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        rec = orig_factory(*args, **kwargs)
        for k, v in _log_extra_fields.items():
            setattr(rec, k, v)
        return rec

    logging.setLogRecordFactory(record_factory)

    root_logger = logging.getLogger()

    def rank0_filter(rec: logging.LogRecord) -> bool:
        if rec.levelno > logging.INFO:
            return True
        return getattr(rec, "global_rank", 0) == 0

    def local_rank0_filter(rec: logging.LogRecord) -> bool:
        if rec.levelno > logging.INFO:
            return True
        return getattr(rec, "local_rank", 0) == 0

    current = getattr(root_logger, "_elevation_log_filter", None)
    if current is not None:
        try:
            root_logger.removeFilter(current)
        except Exception:
            pass

    new_filter = None
    if filter_type == LogFilterType.rank0_only:
        new_filter = rank0_filter
    elif filter_type == LogFilterType.local_rank0_only:
        new_filter = local_rank0_filter
    elif filter_type == LogFilterType.all_ranks:
        new_filter = None
    else:
        raise ValueError(str(filter_type))

    if new_filter is not None:
        root_logger.addFilter(new_filter)
    setattr(root_logger, "_elevation_log_filter", new_filter)

    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

# Exception hook integration

def excepthook(exc_type: type, value: Exception, tb: Any) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, value, tb)
    elif issubclass(exc_type, CliError):
        Console().print(f"[yellow]{value}[/]")
    else:
        log.critical("Unhandled %s: %s", exc_type.__name__, value, exc_info=(exc_type, value, tb))


def install_excepthook() -> None:
    sys.excepthook = excepthook

# Warning filters

def filter_warnings() -> None:
    patterns = ["torch.distributed.*", "TypedStorage is deprecated.*", "failed to load.*"]
    for pat in patterns:
        warnings.filterwarnings("ignore", message=pat)

# Environment variables

def set_env_vars() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# CLI environment preparer

def prepare_cli_env(filter_type: LogFilterType = None) -> None:
    if filter_type is None:
        filter_type = LogFilterType(os.environ.get("LOG_FILTER_TYPE", LogFilterType.rank0_only))
    setup_logging(filter_type)
    install_excepthook()
    filter_warnings()
    set_env_vars()

# Arg cleanup

def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    return f"{name.strip('-').replace('-', '_')}={val}"

# Utilities

def wait_for(cond: Callable[[], bool], desc: str, timeout: float = 10.0) -> None:
    start = time.monotonic()
    while not cond():
        time.sleep(0.5)
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"{desc} timed out")


def is_url(path: PathOrStr) -> bool:
    return bool(re.match(r"[a-z]+://", str(path)))


def dir_is_empty(path: PathOrStr) -> bool:
    p = Path(path)
    return not p.is_dir() or not any(p.iterdir())


def get_progress_bar() -> Progress:
    from cached_path import get_download_progress
    return get_download_progress()


def resource_path(folder: PathOrStr, fname: str, cache: PathOrStr = None, progress: Progress = None) -> Path:
    local = Path(cache or "") / fname
    if cache and local.is_file():
        log.info("Using cache %s", local)
        return local
    from cached_path import cached_path
    return cached_path(f"{folder.rstrip('/')}/{fname}", progress=progress)

# --- Elevation Model Additions and Improvements ---

# Distributed context utilities (stub, customize as needed)
def get_global_rank() -> int:
    return int(os.environ.get("GLOBAL_RANK", 0))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK", 0))

def is_distributed() -> bool:
    return bool(int(os.environ.get("DISTRIBUTED", 0)))

# Remote file operations (GCS, S3, HTTP, Weka, R2)
def _gcs_is_retriable(exc: Exception) -> bool:
    return gcs_is_transient_error(exc) or isinstance(exc, requests.exceptions.ReadTimeout)

_gcs_retry = GCSRetry(predicate=_gcs_is_retriable, initial=1.0, maximum=10.0, multiplier=2.0, timeout=600.0)

@cache
def _get_gcs_client():
    from google.cloud import storage as gcs
    return gcs.Client()

def _gcs_upload(src: Path, bucket: str, key: str, overwrite: bool = False):
    client = _get_gcs_client()
    blob = client.bucket(bucket).blob(key)
    if not overwrite and blob.exists():
        raise FileExistsError(f"gs://{bucket}/{key} exists. Use overwrite=True.")
    blob.upload_from_filename(str(src), retry=_gcs_retry)

def _gcs_file_size(bucket: str, key: str) -> int:
    from google.api_core.exceptions import NotFound
    blob = _get_gcs_client().bucket(bucket).blob(key)
    try:
        blob.reload(retry=_gcs_retry)
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket}/{key}")
    return blob.size

def _gcs_get_bytes_range(bucket: str, key: str, start: int, length: int) -> bytes:
    from google.api_core.exceptions import NotFound
    blob = _get_gcs_client().bucket(bucket).blob(key)
    try:
        return blob.download_as_bytes(start=start, end=start+length-1, retry=_gcs_retry)
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket}/{key}")

def _gcs_find_latest_checkpoint(bucket: str, prefix: str) -> Optional[str]:
    if not prefix.endswith("/"):
        prefix += "/"
    client = _get_gcs_client()
    bucket_obj = client.bucket(bucket)
    suffix = "/config.yaml"
    latest_step, latest_ckpt = None, None
    for blob in bucket_obj.list_blobs(prefix=prefix):
        if not blob.name.endswith(suffix) or blob.size <= 0:
            continue
        name = blob.name[len(prefix):-len(suffix)]
        if "/" in name or not name.startswith("step"):
            continue
        step = int(name[4:].replace("-unsharded", "")) if name[4:] else 0
        if latest_step is None or step > latest_step or (step == latest_step and latest_ckpt and latest_ckpt.endswith("-unsharded")):
            latest_step, latest_ckpt = step, f"gs://{bucket}/{blob.name[:-len(suffix)]}"
    return latest_ckpt

# S3/R2/Weka profile and endpoint utilities
def _get_s3_profile(scheme: str) -> Optional[str]:
    env_map = {"s3": "S3_PROFILE", "r2": "R2_PROFILE", "weka": "WEKA_PROFILE"}
    env = env_map.get(scheme)
    val = os.environ.get(env) if env else None
    if scheme in ("r2", "weka") and not val:
        raise EnvironmentError(f"Missing profile for {scheme}")
    return val

def _get_s3_endpoint(scheme: str) -> Optional[str]:
    env_map = {"r2": "R2_ENDPOINT_URL", "weka": "WEKA_ENDPOINT_URL"}
    return os.environ.get(env_map.get(scheme)) if scheme in env_map else None

@cache
def _get_s3_client(scheme: str):
    return boto3.Session(profile_name=_get_s3_profile(scheme)).client(
        "s3", endpoint_url=_get_s3_endpoint(scheme), config=Config(retries={"max_attempts": 10, "mode": "standard"}), use_ssl=not int(os.environ.get("NO_SSL", "0"))
    )

def _wait_before_retry(attempt: int):
    time.sleep(min(0.5 * 2**attempt, 3.0))

def _s3_upload(src: Path, scheme: str, bucket: str, key: str, overwrite: bool = False, max_attempts: int = 3):
    err = None
    if not overwrite:
        for attempt in range(1, max_attempts+1):
            try:
                _get_s3_client(scheme).head_object(Bucket=bucket, Key=key)
                raise FileExistsError(f"{scheme}://{bucket}/{key} exists. Use overwrite=True.")
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    err = None
                    break
                err = e
            if attempt < max_attempts:
                log.warning("%s failed attempt %d: %s", _s3_upload.__name__, attempt, err)
                _wait_before_retry(attempt)
        if err:
            raise NetworkError(f"Failed to check object existence for {scheme}") from err
    try:
        _get_s3_client(scheme).upload_file(str(src), bucket, key)
    except boto_exceptions.ClientError as e:
        raise NetworkError(f"Failed to upload to {scheme}") from e

def _s3_file_size(scheme: str, bucket: str, key: str, max_attempts: int = 3) -> int:
    err = None
    for attempt in range(1, max_attempts+1):
        try:
            return _get_s3_client(scheme).head_object(Bucket=bucket, Key=key)["ContentLength"]
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"{scheme}://{bucket}/{key}") from e
            err = e
        if attempt < max_attempts:
            log.warning("%s failed attempt %d: %s", _s3_file_size.__name__, attempt, err)
            _wait_before_retry(attempt)
    raise NetworkError(f"Failed to get {scheme} file size") from err

def _s3_get_bytes_range(scheme: str, bucket: str, key: str, start: int, length: int, max_attempts: int = 3) -> bytes:
    err = None
    for attempt in range(1, max_attempts+1):
        try:
            return _get_s3_client(scheme).get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{start+length-1}")["Body"].read()
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"{scheme}://{bucket}/{key}") from e
            err = e
        except (boto_exceptions.HTTPClientError, boto_exceptions.ConnectionError) as e:
            err = e
        if attempt < max_attempts:
            log.warning("%s failed attempt %d: %s", _s3_get_bytes_range.__name__, attempt, err)
            _wait_before_retry(attempt)
    raise NetworkError(f"Failed to get bytes range from {scheme}") from err

def _s3_find_latest_checkpoint(scheme: str, bucket: str, prefix: str) -> Optional[str]:
    if not prefix.endswith("/"):
        prefix += "/"
    resp = _get_s3_client(scheme).list_objects(Bucket=bucket, Prefix=prefix, Delimiter="/")
    assert not resp.get("IsTruncated", False)
    latest_step, latest_ckpt = 0, None
    for item in resp.get("CommonPrefixes", []):
        pfx = item["Prefix"].strip("/")
        ckpt_name = os.path.split(pfx)[-1]
        if not ckpt_name.startswith("step"):
            continue
        try:
            step = int(ckpt_name.replace("step", "").replace("-unsharded", ""))
        except ValueError:
            continue
        try:
            _s3_file_size(scheme, bucket, f"{pfx}/config.yaml")
        except FileNotFoundError:
            continue
        if step > latest_step or (step == latest_step and not ckpt_name.endswith("-unsharded")):
            latest_step, latest_ckpt = step, f"{scheme}://{bucket}/{pfx}"
    return latest_ckpt

def _http_file_size(scheme: str, host: str, path: str) -> int:
    resp = requests.head(f"{scheme}://{host}/{path}", allow_redirects=True)
    return int(resp.headers.get("content-length", 0))

def _http_get_bytes_range(scheme: str, host: str, path: str, start: int, length: int) -> bytes:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{scheme}://{host}/{path}", headers={"Range": f"bytes={start}-{start+length-1}"})
            if len(resp.content) == length:
                return resp.content
            log.warning(f"Expected {length} bytes, got {len(resp.content)}. Retrying...")
        except requests.exceptions.RequestException as e:
            log.warning(f"Attempt {attempt+1}/{max_retries}. Network error: {e}. Retrying...")
        time.sleep(2**attempt)
    raise ValueError(f"Failed to download {length} bytes from {scheme}://{host}/{path} after {max_retries} attempts.")

# Unified file_size, upload, get_bytes_range, find_latest_checkpoint for local/remote
def file_size(path: PathOrStr) -> int:
    if is_url(path):
        from urllib.parse import urlparse
        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_file_size(parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("http", "https"):
            return _http_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return file_size(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    return os.stat(path).st_size

def upload(src: PathOrStr, dst: str, overwrite: bool = False) -> None:
    from urllib.parse import urlparse
    src = Path(src)
    assert src.is_file()
    parsed = urlparse(dst)
    if parsed.scheme == "gs":
        _gcs_upload(src, parsed.netloc, parsed.path.strip("/"), overwrite=overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(src, parsed.scheme, parsed.netloc, parsed.path.strip("/"), overwrite=overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")

def get_bytes_range(src: PathOrStr, start: int, length: int) -> bytes:
    if is_url(src):
        from urllib.parse import urlparse
        parsed = urlparse(str(src))
        if parsed.scheme == "gs":
            return _gcs_get_bytes_range(parsed.netloc, parsed.path.strip("/"), start, length)
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_get_bytes_range(parsed.scheme, parsed.netloc, parsed.path.strip("/"), start, length)
        elif parsed.scheme in ("http", "https"):
            return _http_get_bytes_range(parsed.scheme, parsed.netloc, parsed.path.strip("/"), start, length)
        elif parsed.scheme == "file":
            return get_bytes_range(str(src).replace("file://", "", 1), start, length)
        else:
            raise NotImplementedError(f"get bytes range not implemented for '{parsed.scheme}' files")
    with open(src, "rb") as f:
        f.seek(start)
        return f.read(length)

def find_latest_checkpoint(dir: PathOrStr) -> Optional[PathOrStr]:
    if is_url(dir):
        from urllib.parse import urlparse
        parsed = urlparse(str(dir))
        if parsed.scheme == "gs":
            return _gcs_find_latest_checkpoint(parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_find_latest_checkpoint(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return find_latest_checkpoint(str(dir).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"find_latest_checkpoint not implemented for '{parsed.scheme}' files")
    latest_step, latest_ckpt = 0, None
    for path in Path(dir).glob("step*"):
        if path.is_dir():
            try:
                step = int(path.name.replace("step", "").replace("-unsharded", ""))
            except ValueError:
                continue
            if step > latest_step or (step == latest_step and not path.name.endswith("-unsharded")):
                latest_step, latest_ckpt = step, path
    return latest_ckpt

# HuggingFace dataset utilities
def save_hf_dataset_to_disk(dataset: Union[datasets.DatasetDict, datasets.Dataset], hf_path: str, name: Optional[str], split: str, datasets_dir: PathOrStr) -> None:
    """Save a HuggingFace dataset to disk under datasets_dir."""
    dataset_path = Path(datasets_dir) / hf_path / (name or "none") / split
    dataset.save_to_disk(str(dataset_path))

def load_hf_dataset(path: str, name: Optional[str], split: str):
    """Load a HuggingFace dataset from disk."""
    dataset_rel_path = os.path.join("hf_datasets", path, name or "none", split)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"HF dataset {path} name {name} split {split} not found in {dataset_rel_path}")
        return datasets.load_from_disk(str(dataset_path))

def load_oe_eval_requests(path: str, name: Optional[str] = None, split: Optional[str] = None):
    """Load OE-eval request file from data/oe_eval_tasks."""
    dataset_rel_path = os.path.join("oe_eval_tasks", path)
    if name:
        dataset_rel_path = os.path.join(dataset_rel_path, name)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"OE Eval dataset not found in {dataset_rel_path}")
        data_file = dataset_path / "requests.jsonl.gz"
        if not data_file.is_file():
            data_file = dataset_path / "requests.jsonl"
        if not data_file.is_file():
            raise FileNotFoundError(f"OE Eval dataset file requests-{split}.jsonl(.gz) missing in {dataset_rel_path}")
        requests_list = []
        if data_file.suffix == ".gz":
            with gzip.open(data_file, "r") as f:
                for line in f:
                    requests_list.append(json.loads(line.decode("utf-8").strip()))
        else:
            with open(data_file, "r") as f:
                for line in f:
                    requests_list.append(json.loads(line.strip()))
        config = None
        config_file = dataset_path / "config.json"
        if config_file.is_file():
            with open(config_file, "r") as f:
                config = json.load(f)
        return config, requests_list

# Default thread count utility
def default_thread_count() -> int:
    return int(os.environ.get("NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))

# Pass-through function
def pass_through_fn(fn: Callable, *args, **kwargs):
    return fn(*args, **kwargs)

# --- End Elevation Model Additions ---
