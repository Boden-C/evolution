from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .aliases import PathOrStr
from .exceptions import ConfigurationError
from .util import StrEnum

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class BaseConfig:
    """
    Shared base config for elevation model configs. Handles YAML load/save, legacy update, asdict, update_with, and Omegaconf support.
    """

    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        def path_choose(*paths) -> str:
            from .util import is_url

            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        def path_last_checkpoint(path) -> str:
            from .util import find_latest_checkpoint

            latest_checkpoint = find_latest_checkpoint(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(f"Could not find a latest checkpoint at {path}")
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver("path.last_checkpoint", path_last_checkpoint, replace=True)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        return config

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def update_with(self, **kwargs):
        result = deepcopy(self)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"


class BlockType(StrEnum):
    sequential = "sequential"
    llama = "llama"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    normal = "normal"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"


@dataclass
class ModelConfig(BaseConfig):
    """
    Elevation model configuration.
    """

    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    clip_qkv: Optional[float] = None
    n_layers: int = 12
    mlp_ratio: int = 4
    mlp_hidden_size: Optional[int] = None
    activation_type: ActivationType = ActivationType.swiglu
    block_type: BlockType = BlockType.sequential
    block_group_size: int = 1
    alibi: bool = False
    alibi_bias_max: float = 8.0
    rope: bool = False
    rope_full_precision: bool = True
    rope_theta: int = 10000
    flash_attention: bool = False
    attention_dropout: float = 0.1
    multi_query_attention: Optional[bool] = None
    attention_layer_norm: bool = False
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    embedding_layer_norm: bool = False
    layer_norm_type: LayerNormType = LayerNormType.default
    layer_norm_with_affine: bool = True
    layer_norm_eps: float = 1e-5
    attention_layer_norm_with_affine: bool = True
    max_sequence_length: int = 1024
    include_bias: bool = True
    bias_for_layer_norm: Optional[bool] = None
    scale_logits: bool = False
    vocab_size: int = 50257
    embedding_size: Optional[int] = 50304
    weight_tying: bool = True
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    init_device: Optional[str] = None
    init_fn: InitFnType = InitFnType.normal
    init_std: float = 0.02
    init_cutoff_factor: Optional[float] = None
    precision: Optional[str] = None
    scale_emb_init: bool = False
    emb_init_std: Optional[float] = None
    norm_after: bool = False

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ConfigurationError("d_model must be divisible by n_heads")
        if self.max_sequence_length <= 0:
            raise ConfigurationError("max_sequence_length must be > 0")

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise ConfigurationError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.lionw
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5
    no_decay_norm_and_bias: Optional[bool] = None
    selective_updates: bool = False
    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    metrics_log_interval: Optional[int] = None
    record_update_metrics: bool = False

    def __post_init__(self):
        self.betas = tuple(self.betas)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)
            if hasattr(new_config, "name") and new_config.name == "decoupled_lionw":
                new_config.name = "lionw"
                if hasattr(new_config, "eps"):
                    del new_config.eps
        return new_config


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"
    cosine_linear_envelope = "cosine_linear_envelope"
    constant_with_warmup = "constant_with_warmup"


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1
    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    grad_clip_warmup_factor: Optional[float] = None
    warmup_min_lr: Optional[float] = None


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class InstanceFilterConfig(BaseConfig):
    repetition_max_period: int = 13
    repetition_min_period: int = 1
    repetition_max_count: int = 32


@dataclass
class DataConfig(BaseConfig):
    paths: Optional[List[str]] = None
    memmap_dtype: str = "uint16"
    datasets: Optional[Dict[str, List[str]]] = None
    label_mask_paths: Optional[List[str]] = None
    pad_direction: PaddingDirection = PaddingDirection.right
    generate_attention_mask: bool = False
    generate_doc_lengths: bool = False
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0
    seed: Optional[int] = None
    instance_filter: Optional[InstanceFilterConfig] = None
    custom_dataset: Optional['CustomDatasetConfig'] = None

    @property
    def effective_memmap_dtype(self):
        try:
            np.dtype(dtype := getattr(np, self.memmap_dtype))
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Value {self.memmap_dtype} is not a valid numpy type") from e
        return dtype


@dataclass
class CustomDatasetCollatorConfig(BaseConfig):
    input_id_field: str = "input_ids"
    attention_mask_field: Optional[str] = None
    attention_bias_field: Optional[str] = None
    label_mask_field: Optional[str] = None
    index_field: Optional[str] = None
    instance_mask_field: Optional[str] = None
    doc_lens_field: Optional[str] = None
    metadata_field: Optional[str] = None


@dataclass
class CustomDatasetConfig(BaseConfig):
    name: str
    module: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    collate_fn: Optional[str] = None
    token_field: Optional[str] = None
    collate_config: Optional[CustomDatasetCollatorConfig] = field(default_factory=CustomDatasetCollatorConfig)


class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


@dataclass
class EvaluatorConfig(BaseConfig):
    label: str
    type: EvaluatorType = EvaluatorType.lm
    data: DataConfig = field(default_factory=DataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    truncate_direction: TruncationDirection = TruncationDirection.right


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "ai2-llm"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = None
    fullgraph: bool = False
    backend: str = "inductor"
    dynamic: Optional[bool] = None


class DistributedStrategy(StrEnum):
    ddp = "ddp"
    fsdp = "fsdp"
    single = "single"


class DDPGradSyncMode(StrEnum):
    batch = "batch"
    micro_batch = "micro_batch"


@dataclass
class DDPConfig(BaseConfig):
    grad_sync_mode: DDPGradSyncMode = DDPGradSyncMode.batch
    find_unused_params: bool = False


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    by_block_and_size = "by_block_and_size"
    by_block_group = "by_block_group"
    by_block_group_and_size = "by_block_group_and_size"
    size_based = "size_based"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    mixed = "mixed"


@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    wrapping_strategy: Optional[FSDPWrapStrategy] = None
    precision: Optional[FSDPPrecision] = FSDPPrecision.pure
    hybrid_sharding_num_model_replicas: Optional[int] = None


@dataclass
class SingleGPUConfig(BaseConfig):
    device: str = "auto"

    def get_device(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif self.device == "mps" and not torch.backends.mps.is_available():
            raise ConfigurationError("MPS not available.")
        elif self.device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA not available.")
        else:
            return torch.device(self.device)


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"
    elevation_core = "elevation_core"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_eight = "one_in_eight"
    two_in_three = "two_in_three"
    three_in_four = "three_in_four"
    fine_grained = "fine_grained"


@dataclass
class TrainConfig(BaseConfig):
    run_name: Optional[str] = None
    seed: int = 6198
    epoch: Optional[int] = None
    dry_run: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    restore_dataloader: bool = True
    fast_forward_batches: Optional[int] = None
    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    eval_interval: int = 1000
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    save_folder: str = "./"
    remote_save_folder: Optional[str] = None
    canceled_check_interval: int = 50
    save_interval: Optional[int] = 1000
    save_interval_unsharded: Optional[int] = None
    save_interval_ephemeral: Optional[int] = None
    save_num_checkpoints_to_keep: int = -1
    save_num_unsharded_checkpoints_to_keep: int = -1
    save_overwrite: bool = False
    force_save_unsharded: bool = False
    no_pre_train_checkpoint: bool = False
    load_path: Optional[str] = None
    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    try_load_latest_save: bool = False
    reset_optimizer_state: bool = False
    reset_trainer_state: bool = False
    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    new_style_checkpoints: Optional[bool] = None
    max_duration: Union[int, str] = 10000
    global_train_batch_size: int = 512
    device_train_batch_size: Optional[int] = None
    device_train_microbatch_size: int = 16
    device_eval_batch_size: int = 16
    eval_subset_num_batches: int = -1
    eval_on_load: bool = False
    device_train_grad_accum: Optional[int] = None
    max_grad_norm: Optional[float] = None
    max_grad_norm_ratio: Optional[float] = None
    precision: Optional[str] = None
    wandb: Optional[WandbConfig] = None
    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    console_log_interval: int = 1
    gen1_gc_interval: Optional[int] = 1
    compile: Optional[CompilerConfig] = None
    distributed_strategy: Optional[DistributedStrategy] = DistributedStrategy.fsdp
    fsdp: Optional[FSDPConfig] = field(default_factory=FSDPConfig)
    ddp: Optional[DDPConfig] = None
    single: SingleGPUConfig = field(default_factory=lambda: SingleGPUConfig(device="auto"))
    softmax_auxiliary_loss: bool = False
    auxiliary_loss_multiplier: Optional[float] = 1e-4
    time_limit: Optional[float] = None
    extra_steps_after_cancel: int = 10
    early_stopping_factor: Optional[float] = None
    save_data_indices: bool = True
    python_profiling: bool = False
    torch_profiling: bool = False
    stop_at: Optional[int] = None
    stop_after: Optional[int] = None
    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    fused_loss: Optional[bool] = None
    hf_datasets_cache_dir: Optional[str] = None
    module_outputs_save_steps: Optional[List[int]] = None

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> Optional[MixedPrecision]:
        if self.fsdp is not None:
            if self.fsdp.precision is None:
                return None
            elif self.fsdp.precision == FSDPPrecision.pure:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=self.autocast_precision,
                    buffer_dtype=self.autocast_precision,
                )
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
            else:
                raise NotImplementedError(f"{self.fsdp.precision}")
        else:
            raise ValueError("self.fsdp is None!")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)
            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    new_config.activation_checkpointing = ActivationCheckpointingStrategy.whole_layer
            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(new_config.optimizer)
        return new_config


__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "InitFnType",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "InstanceFilterConfig",
    "EvaluatorConfig",
    "TokenizerConfig",
    "TrainConfig",
    "PaddingDirection",
    "TruncationDirection",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "DDPConfig",
    "DistributedStrategy",
    "DDPGradSyncMode",
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "SingleGPUConfig",
    "CheckpointType",
]
