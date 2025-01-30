from __future__ import annotations

import importlib
from typing import Iterator, List, Optional, Tuple


class InMemoryTextDataset:
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - simple
        return iter(self.texts)


def extract_module_and_class(name: str) -> Tuple[Optional[str], str]:
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def build_custom_dataset(name: str, *, module: Optional[str] = None, **kwargs):
    mod, cls_name = extract_module_and_class(name)
    module = module or mod
    if module is None:
        raise ValueError("Module must be provided if class name is not fully qualified")
    try:
        mod_obj = importlib.import_module(module)
        cls = getattr(mod_obj, cls_name)
    except (ImportError, AttributeError) as e:  # pragma: no cover - import guard
        raise ImportError(f"Could not import {cls_name} from {module}: {e}") from e
    return cls(**kwargs)


__all__ = ["InMemoryTextDataset", "extract_module_and_class", "build_custom_dataset"]
