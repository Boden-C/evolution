from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Callable, Union
import re
import time
from pathlib import Path
from enum import Enum

class StrEnum(str, Enum):
    """
    Equivalent to Python's enum.StrEnum (since 3.11). Compatibility for older versions.
    """
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"

def dataclass_asdict_shallow(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}

def flatten_dict(dictionary: Dict[str, Any], parent_key: str = "", separator: str = ".", include_lists: bool = False) -> Dict[str, Any]:
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

def clean_opt(arg: str) -> str:
    if "=" not in arg:
        return arg.strip('-').replace('-', '_')
    name, val = arg.split("=", 1)
    return f"{name.strip('-').replace('-', '_')}={val}"

def wait_for(cond: Callable[[], bool], desc: str, timeout: float = 10.0) -> None:
    start = time.monotonic()
    while not cond():
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {desc}")
        time.sleep(0.1)

def is_url(path: Union[str, Path]) -> bool:
    return bool(re.match(r"[a-z]+://", str(path)))

def dir_is_empty(path: Union[str, Path]) -> bool:
    p = Path(path)
    return not p.is_dir() or not any(p.iterdir())

def default_thread_count() -> int:
    import os
    return max(1, os.cpu_count() or 1)

def pass_through_fn(fn: Callable, *args, **kwargs):
    return fn(*args, **kwargs)

def resource_path(folder: Union[str, Path], fname: str, cache: Union[str, Path] = None, progress: Any = None) -> Path:
    local = Path(cache or "") / fname
    if cache and local.is_file():
        return local
    from cached_path import cached_path
    return Path(cached_path(f"{str(folder).rstrip('/')}/{fname}", progress=progress))
