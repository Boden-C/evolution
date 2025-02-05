from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Callable, Any
import gzip
import json
from urllib.parse import urlparse
import os
import re
import requests
from functools import lru_cache as cache
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Union, Optional

from .exceptions import NetworkError, EnvironmentError
from .compatability.data import get_data_path
import datasets

# --- Remote/local file operation helpers ---
def _gcs_file_size(bucket: str, key: str) -> int:
    raise NotImplementedError(
        "Cannot determine GCS object size without the Google Cloud Storage SDK and credentials. "
        "Install 'google-cloud-storage' and configure authentication (e.g., GOOGLE_APPLICATION_CREDENTIALS) to enable."
    )

def _s3_file_size(scheme: str, bucket: str, key: str) -> int:
    raise NotImplementedError(
        f"Cannot determine {scheme.upper()} object size without an appropriate SDK (e.g., boto3) and cloud credentials. "
        "Add the provider SDK and configure auth to enable."
    )

def _http_file_size(scheme: str, host: str, path: str) -> int:
    # Try HEAD first for Content-Length, then fall back to a ranged GET parsing Content-Range
    url = f"{scheme}://{host}/{path.lstrip('/')}"
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            return int(cl)
        # Fallback: request a single byte to parse total size from Content-Range
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=15)
        if resp.status_code in (200, 206):
            cr = resp.headers.get("Content-Range")
            if cr:
                # Format: bytes start-end/total
                total = cr.split("/")[-1]
                if total.isdigit():
                    return int(total)
        # If still unknown, we can't reliably determine without downloading
        raise NetworkError(f"Could not determine HTTP file size for {url} (no Content-Length)")
    except requests.RequestException as e:
        raise NetworkError(f"HTTP file size request failed for {url}: {e}") from e

def _gcs_upload(src: Path, bucket: str, key: str, overwrite: bool = False):
    raise NotImplementedError(
        "GCS upload requires the Google Cloud Storage SDK and authenticated client; omitted to avoid cloud-side effects and extra deps."
    )

def _s3_upload(src: Path, scheme: str, bucket: str, key: str, overwrite: bool = False):
    raise NotImplementedError(
        f"Upload to {scheme.upper()} requires a provider SDK (e.g., boto3 for s3) and credentials; disabled here to avoid side effects and optional deps."
    )

def _gcs_get_bytes_range(bucket: str, key: str, start: int, length: int) -> bytes:
    raise NotImplementedError(
        "Byte-range reads from GCS require the google-cloud-storage client and authentication; not available in this runtime."
    )

def _s3_get_bytes_range(scheme: str, bucket: str, key: str, start: int, length: int) -> bytes:
    raise NotImplementedError(
        f"Byte-range reads from {scheme.upper()} require an SDK (e.g., boto3) and credentials; not available in this runtime."
    )

def _http_get_bytes_range(scheme: str, host: str, path: str, start: int, length: int) -> bytes:
    url = f"{scheme}://{host}/{path.lstrip('/')}"
    end = start + length - 1
    try:
        resp = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=30)
        # Some servers may ignore Range and return 200 with full content; slice if needed
        if resp.status_code in (200, 206):
            data = resp.content
            return data[:length]
        resp.raise_for_status()
        return b""
    except requests.RequestException as e:
        raise NetworkError(f"HTTP ranged GET failed for {url}: {e}") from e

def _gcs_find_latest_checkpoint(bucket: str, prefix: str) -> Optional[str]:
    raise NotImplementedError(
        "Listing GCS prefixes to find checkpoints requires google-cloud-storage and auth; not available here."
    )

def _s3_find_latest_checkpoint(scheme: str, bucket: str, prefix: str) -> Optional[str]:
    raise NotImplementedError(
        f"Finding latest checkpoint on {scheme.upper()} requires a provider SDK (e.g., boto3) and credentials; not available here."
    )

def is_url(path: Union[str, Path]) -> bool:
    return bool(re.match(r"[a-z]+://", str(path)))

def dir_is_empty(path: Union[str, Path]) -> bool:
    p = Path(path)
    return not p.is_dir() or not any(p.iterdir())

def file_size(path: Union[str, Path]) -> int:
    if is_url(path):
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

def upload(src: Union[str, Path], dst: str, overwrite: bool = False) -> None:
    parsed = urlparse(dst)
    src = Path(src)
    assert src.is_file()
    if parsed.scheme == "gs":
        _gcs_upload(src, parsed.netloc, parsed.path.strip("/"), overwrite=overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(src, parsed.scheme, parsed.netloc, parsed.path.strip("/"), overwrite=overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")

def get_bytes_range(src: Union[str, Path], start: int, length: int) -> bytes:
    if is_url(src):
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
            raise NotImplementedError(f"get_bytes_range not implemented for '{parsed.scheme}' files")
    with open(src, "rb") as f:
        f.seek(start)
        return f.read(length)

def find_latest_checkpoint(dir: Union[str, Path]) -> Optional[Union[str, Path]]:
    if is_url(dir):
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
                step = int(str(path.name).replace("step", "").replace("-unsharded", ""))
            except ValueError:
                continue
            if step > latest_step:
                latest_step = step
                latest_ckpt = path
    return latest_ckpt

def dataclass_asdict_shallow(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to a shallow dict or wrap a value into {"value": obj}."""
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
    """Flatten a nested dictionary into a single-level mapping.

    Lists are optionally converted into integer-keyed mappings.
    """
    items: Dict[str, Any] = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, separator, include_lists))
        elif include_lists and isinstance(value, list):
            for idx, item in enumerate(value):
                items.update(flatten_dict({str(idx): item}, new_key, separator, include_lists))
        else:
            items[new_key] = value
    return items


def save_hf_dataset_to_disk(dataset: Any, hf_path: str, name: Optional[str], split: str, datasets_dir: Union[str, None]) -> None:
    """Save a HuggingFace dataset to disk."""
    dataset_path = Path(datasets_dir) / hf_path / (name or "none") / split
    return dataset.save_to_disk(str(dataset_path))


def load_hf_dataset(path: str, name: Optional[str], split: str):
    """Load a HuggingFace dataset from disk."""
    dataset_rel_path = os.path.join("hf_datasets", path, name or "none", split)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(
                f"HF dataset {path} name {name} split {split} not found in directory {dataset_rel_path}"
            )
        return datasets.load_from_disk(str(dataset_path))


def load_oe_eval_requests(path: str, name: Optional[str] = None, split: Optional[str] = None):
    """Load OE evaluation requests from disk."""
    dataset_rel_path = os.path.join("oe_eval_tasks", path)
    if name is not None:
        dataset_rel_path = os.path.join(dataset_rel_path, name)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"OE Eval dataset not found in directory {dataset_rel_path}")
        data_file = dataset_path / "requests.jsonl.gz"
        if not data_file.is_file():
            data_file = dataset_path / "requests.jsonl"
        if not data_file.is_file():
            raise FileNotFoundError(
                f"OE Eval dataset file requests-{split}.jsonl(.gz) missing in directory {dataset_rel_path}"
            )
        requests = []
        if data_file.suffix == ".gz":
            with gzip.open(data_file, "r") as file:
                for line in file:
                    requests.append(json.loads(line.decode("utf-8").strip()))
        else:
            with open(data_file, "r") as file:
                for line2 in file:
                    requests.append(json.loads(line2.strip()))
        config = None
        config_file = dataset_path / "config.json"
        if config_file.is_file():
            with open(config_file, "r") as file:
                config = json.load(file)
        return config, requests
