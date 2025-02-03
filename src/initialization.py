"""
Initializes weights and biases for supported modules with configurable options, logging, and extensibility.
"""

from __future__ import annotations

from typing import Optional, Union, Sequence

import torch
import torch.nn as nn

from src.config import get_config  # Project config integration
from src.logging import log_event  # Project logging integration

__all__ = ["initialize_module"]


def initialize_module(
    module: Union[nn.Linear, nn.Embedding, nn.Conv2d, Sequence[nn.Module]],
    std: float = 0.02,
    cutoff_factor: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    registry_callback: Optional[callable] = None,
) -> None:
    """
    Initialize weights and biases for supported modules.
    Args:
        module: Module or sequence of modules to initialize.
        std: Standard deviation for normal initialization.
        cutoff_factor: Truncation cutoff factor for truncated normal.
        seed: Optional random seed for reproducibility.
        device: Optional device placement.
        dry_run: If True, only log actions without modifying modules.
        registry_callback: Optional callback for model registry/tracking.
    """
    config = get_config()
    if seed is not None:
        torch.manual_seed(seed)
    modules = module if isinstance(module, Sequence) else [module]
    for mod in modules:
        try:
            if device is not None:
                mod.to(device)
            if cutoff_factor is not None:
                cutoff_value = cutoff_factor * std
                if not dry_run:
                    nn.init.trunc_normal_(mod.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
                log_event(f"Initialized {type(mod).__name__} weights with truncated normal std={std}, cutoff={cutoff_value}")
            else:
                if not dry_run:
                    nn.init.normal_(mod.weight, mean=0.0, std=std)
                log_event(f"Initialized {type(mod).__name__} weights with normal std={std}")
            if hasattr(mod, "bias") and mod.bias is not None:
                if not dry_run:
                    nn.init.zeros_(mod.bias)
                log_event(f"Initialized {type(mod).__name__} bias to zeros")
            if registry_callback:
                registry_callback(mod)
        except Exception as exc:
            log_event(f"Initialization failed for {type(mod).__name__}: {exc}", level="error")
            raise


# Extensibility hook for custom initializers
def register_custom_initializer(module_type: type, initializer: callable) -> None:
    """
    Register a custom initializer for a module type.
    Args:
        module_type: Type of module to register.
        initializer: Callable initializer.
    """
    # ...implementation for registry...
    pass
