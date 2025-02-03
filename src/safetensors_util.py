from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Mapping, Any

import torch
try:
    from safetensors.torch import load_file, save_file
except ImportError:
    load_file = None
    save_file = None

from src.config import PathOrStr  # Project-specific type
from src.logging import log  # Project-specific logging

__all__ = [
    "state_dict_to_safetensors_file",
    "safetensors_file_to_state_dict",
    "check_safetensors_metadata"
]

@dataclass(eq=True, frozen=True)
class SafetensorsKey:
    keys: Tuple
    value_is_pickled: bool

def encode_safetensors_key(key: SafetensorsKey) -> str:
    """Encode a SafetensorsKey for safe serialization."""
    b = pickle.dumps((key.keys, key.value_is_pickled))
    b = base64.urlsafe_b64encode(b)
    return str(b, "ASCII")

def decode_safetensors_key(key: str) -> SafetensorsKey:
    """Decode a serialized SafetensorsKey."""
    b = base64.urlsafe_b64decode(key)
    keys, value_is_pickled = pickle.loads(b)
    return SafetensorsKey(keys, value_is_pickled)

def flatten_state_dict(d: Dict) -> Dict[SafetensorsKey, torch.Tensor]:
    """Flatten nested dicts and non-tensor values for safetensors serialization."""
    result = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[SafetensorsKey((key,), False)] = value
        elif isinstance(value, dict):
            value = flatten_state_dict(value)
            for inner_key, inner_value in value.items():
                result[SafetensorsKey((key,) + inner_key.keys, inner_key.value_is_pickled)] = inner_value
        else:
            pickled = bytearray(pickle.dumps(value))
            pickled_tensor = torch.frombuffer(pickled, dtype=torch.uint8)
            result[SafetensorsKey((key,), True)] = pickled_tensor
    return result

def unflatten_state_dict(d: Dict[SafetensorsKey, torch.Tensor]) -> Dict:
    """Restore nested dicts and non-tensor values from safetensors serialization."""
    result: Dict = {}
    for key, value in d.items():
        if key.value_is_pickled:
            value = pickle.loads(value.numpy().data)
        target_dict = result
        for k in key.keys[:-1]:
            new_target_dict = target_dict.get(k)
            if new_target_dict is None:
                new_target_dict = {}
                target_dict[k] = new_target_dict
            target_dict = new_target_dict
        target_dict[key.keys[-1]] = value
    return result

def state_dict_to_safetensors_file(state_dict: Dict, filename: PathOrStr) -> None:
    """Serialize a state dict to a safetensors file."""
    if save_file is None:
        log.error("safetensors.torch.save_file not available.")
        raise ImportError("safetensors.torch.save_file not available.")
    try:
        flat_dict = flatten_state_dict(state_dict)
        encoded_dict = {encode_safetensors_key(k): v for k, v in flat_dict.items()}
        save_file(encoded_dict, filename)
        log.info(f"State dict saved to safetensors file: {filename}")
    except Exception as e:
        log.error(f"Failed to save safetensors file: {e}")
        raise

def safetensors_file_to_state_dict(filename: PathOrStr, map_location: Optional[str] = None) -> Dict:
    """Load a state dict from a safetensors file."""
    if load_file is None:
        log.error("safetensors.torch.load_file not available.")
        raise ImportError("safetensors.torch.load_file not available.")
    if map_location is None:
        map_location = "cpu"
    try:
        loaded = load_file(filename, device=map_location)
        decoded = {decode_safetensors_key(k): v for k, v in loaded.items()}
        state_dict = unflatten_state_dict(decoded)
        log.info(f"State dict loaded from safetensors file: {filename}")
        return state_dict
    except Exception as e:
        log.error(f"Failed to load safetensors file: {e}")
        return {}

def check_safetensors_metadata(meta: Dict[str, str]) -> bool:
    """Validate safetensors metadata."""
    return bool(meta)
