from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import ModelConfig
from .exceptions import ConfigurationError
from .util.torch_util import get_default_device


@dataclass
class ElevationOutput:
    logits: "object"  # Deferred torch.Tensor to avoid hard dependency at import


class Elevation:
    """A tiny reference model skeleton with a stable interface.

    This is not a performant model. It demonstrates structure: config validation,
    device management, and a forward contract. It only creates parameters and runs
    if PyTorch is available; otherwise it raises at call time.
    """

    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config
        self.device_str = get_default_device(prefer_cuda=(config.init_device == "cuda"))

        # Lazy torch import and state; keep import-time cheap and safe
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            d_model = config.d_model
            vocab = config.vocab_size
            self._embed = nn.Embedding(vocab, d_model)
            self._lm_head = nn.Linear(d_model, vocab, bias=False)
            self._pos = nn.Embedding(config.max_sequence_length, d_model)
            self._blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
                                          for _ in range(config.n_layers)])
            self.to(torch.device(self.device_str))
        except Exception as exc:  # pragma: no cover - import/runtime guard
            # Defer heavy import errors until first forward
            self._import_error = exc
            self._embed = None
            self._lm_head = None
            self._pos = None
            self._blocks = None

    def to(self, device):  # pragma: no cover - passthrough convenience
        try:
            import torch  # type: ignore

            modules = [self._embed, self._lm_head, self._pos, self._blocks]
            for m in modules:
                if hasattr(m, "to"):
                    m.to(device)
        except Exception:
            pass
        return self

    @property
    def device(self):  # pragma: no cover - simple shim
        try:
            import torch  # type: ignore

            return torch.device(self.device_str)
        except Exception:
            return self.device_str

    def forward(self, input_ids: "object") -> ElevationOutput:
        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore

            if not isinstance(input_ids, torch.Tensor):
                raise ConfigurationError("input_ids must be a torch.Tensor if torch is available")
            x = self._embed(input_ids)
            seq_len = x.size(1)
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self._pos(pos)
            for blk in self._blocks:
                x = x + blk(x)
            logits = self._lm_head(x)
            return ElevationOutput(logits=logits)
        except AttributeError as e:
            raise ConfigurationError("Torch modules not initialized; install torch to run the model") from e


__all__ = ["Elevation", "ElevationOutput"]
