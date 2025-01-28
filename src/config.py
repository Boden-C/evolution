from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .aliases import PathOrStr
from .exceptions import ConfigurationError
from .util import StrEnum


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class LayerNormType(StrEnum):
    layer_norm = "layer_norm"
    rms_norm = "rms_norm"


class BlockType(StrEnum):
    sequential = "sequential"


class OptimizerType(StrEnum):
    adamw = "adamw"
    lionw = "lionw"


class SchedulerType(StrEnum):
    cosine = "cosine"
    linear = "linear"
    constant = "constant"


class DistributedStrategy(StrEnum):
    single = "single"
    ddp = "ddp"
    fsdp = "fsdp"


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50280
    max_sequence_length: int = 2048
    activation: ActivationType = ActivationType.swiglu
    layer_norm: LayerNormType = LayerNormType.rms_norm
    block_type: BlockType = BlockType.sequential
    dropout: float = 0.0
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    weight_tying: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    init_device: str = "cpu"

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ConfigurationError("d_model must be divisible by n_heads")
        if self.max_sequence_length <= 0:
            raise ConfigurationError("max_sequence_length must be > 0")


@dataclass
class OptimizerConfig:
    type: OptimizerType = OptimizerType.adamw
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    type: SchedulerType = SchedulerType.cosine
    warmup_steps: int = 2000
    total_steps: int = 100000


@dataclass
class TokenizerConfig:
    name_or_path: Optional[PathOrStr] = None
    truncation_direction: str = "right"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    seed: int = 1234
    distributed: DistributedStrategy = DistributedStrategy.single
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    save_folder: PathOrStr = "./checkpoints"
    module_outputs_save_steps: Optional[List[int]] = None
    extra: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "ActivationType",
    "LayerNormType",
    "BlockType",
    "OptimizerType",
    "SchedulerType",
    "DistributedStrategy",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TokenizerConfig",
    "TrainConfig",
]
