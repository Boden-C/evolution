"""Run this script with 'torchrun'."""

import gzip
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from src.data import build_train_dataloader
from src.eval import build_evaluators
from src.exceptions import EvolutionCliError, EvolutionConfigurationError
from src.model import Model
from src.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from src.torch_util import (
    SingleAccelerator,
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from src.train import Trainer
from src.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    """
    Main entry point for model training in the evolution project.
    Validates configuration, sets up environment, initializes components, and starts training.
    """
    if cfg.run_name is None:
        raise EvolutionConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "Reset requested for optimizer or trainer state without checkpoint loading. Setting has no effect."
        )

    barrier()

    # Set device.
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Fill configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if getattr(cfg.optimizer, "no_decay_norm_and_bias", None) is not None:
        log.warning(
            "Deprecated config option `no_decay_norm_and_bias` set. Use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise EvolutionConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Start W&B run if enabled.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    barrier()

    # Set seed for reproducibility.
    seed_all(cfg.seed)

    # Construct data loader and evaluators.
    train_loader = build_train_dataloader(cfg)
    evaluators = build_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    log.info("Building model...")
    model = Model(cfg.model)
    log.info(f"Total number of parameters: {model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    # Compile blocks if requested.
    if cfg.compile is not None:
        if cfg.model.block_group_size != 1:
            raise EvolutionConfigurationError("Compile is only supported with block_group_size 1.")
        for block in model.transformer.blocks:
            block.compile(**cfg.compile.asdict())

    model.set_activation_checkpointing(cfg.activation_checkpointing)

    # Distributed strategy setup.
    if cfg.distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")
        assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"
        if cfg.model.init_device != "cuda":
            raise EvolutionConfigurationError("DDP requires init_device to be 'cuda'.")
        if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
            raise EvolutionConfigurationError(
                "`find_unused_params` is True. DDP needs micro_batch grad_sync_mode."
            )
        param_init_fn = None
        dist_model = DDP(model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)
    elif cfg.distributed_strategy == DistributedStrategy.fsdp:
        log.info("Wrapping model with FSDP...")
        assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
        wrap_policy = model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())
            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None
        device_mesh = None
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                raise EvolutionConfigurationError("Hybrid sharding requires torch >= 2.2.0.")
            from torch.distributed.device_mesh import init_device_mesh
            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )
            if num_model_replicas <= 0:
                raise EvolutionConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be positive.")
            if get_world_size() % num_model_replicas != 0:
                raise EvolutionConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size.")
            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
            hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh
        dist_model = FSDP(
            model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )
    elif cfg.distributed_strategy == DistributedStrategy.single:
        param_init_fn = None
        if model is None:
            raise EvolutionConfigurationError("Model initialization failed.")
        model = model.to(device)
        dist_model = SingleAccelerator(model)

    if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
        model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(dist_model)

    # Construct optimizer and scheduler.
    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            raise EvolutionConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Trainer object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=indices_file,
    ) as trainer:
        if cfg.try_load_latest_save:
            checkpoint_dir = None
            if (
                cfg.save_folder is not None
                and (checkpoint_dir := find_latest_checkpoint(cfg.save_folder)) is not None
            ):
                log.info("Setting load path to local checkpoint %s", checkpoint_dir)
                cfg.load_path = str(checkpoint_dir)
            elif (
                cfg.remote_save_folder is not None
                and (checkpoint_dir := find_latest_checkpoint(cfg.remote_save_folder)) is not None
            ):
                log.info("Setting load path to remote checkpoint %s", checkpoint_dir)
                cfg.load_path = str(checkpoint_dir)
            if checkpoint_dir is not None and not cfg.restore_dataloader:
                log.info(
                    "restore_dataloader=False with try_load_latest_save=True. Overwriting previous checkpoints. Setting restore_dataloader=True."
                )
                cfg.restore_dataloader = True
            if checkpoint_dir is not None and cfg.reset_trainer_state:
                log.info(
                    "reset_trainer_state=True with try_load_latest_save=True. Setting reset_trainer_state=False."
                )
                cfg.reset_trainer_state = False
            if checkpoint_dir is not None and cfg.reset_optimizer_state:
                log.info(
                    "reset_optimizer_state=True with try_load_latest_save=True. Setting reset_optimizer_state=False."
                )
                cfg.reset_optimizer_state = False

        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            if cfg.distributed_strategy == DistributedStrategy.ddp:
                checkpoint_type = CheckpointType.unsharded
                if cfg.save_interval_unsharded is None:
                    log.warning(
                        "DDP requires save_interval_unsharded. Using save_interval value."
                    )
                    cfg.save_interval_unsharded = cfg.save_interval
                if cfg.save_num_unsharded_checkpoints_to_keep == 0:
                    log.warning(
                        "DDP requires save_num_unsharded_checkpoints_to_keep. Using save_num_checkpoints_to_keep value."
                    )
                    cfg.save_num_unsharded_checkpoints_to_keep = cfg.save_num_checkpoints_to_keep
            elif cfg.distributed_strategy == DistributedStrategy.fsdp:
                checkpoint_type = (
                    CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
                )
            elif cfg.distributed_strategy == DistributedStrategy.single:
                checkpoint_type = CheckpointType.unsharded
                if cfg.save_interval_unsharded is None:
                    log.warning(
                        "Single accelerator training requires save_interval_unsharded. Using save_interval value."
                    )
                    cfg.save_interval_unsharded = cfg.save_interval
                if cfg.save_num_unsharded_checkpoints_to_keep == 0:
                    log.warning(
                        "Single accelerator training requires save_num_unsharded_checkpoints_to_keep. Using save_num_checkpoints_to_keep value."
                    )
                    cfg.save_num_unsharded_checkpoints_to_keep = cfg.save_num_checkpoints_to_keep
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            log.info(f"Checkpoint saved to {checkpoint_path}")
            log.info("Attempting to load pre-train checkpoint...")
            trainer.restore_checkpoint(
                checkpoint_path, checkpoint_type=checkpoint_type, local_cache=local_checkpoint_cache
            )
            log.info("Checkpoint successfully loaded")

        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(
                cfg.load_path,
                load_optimizer_state=not cfg.reset_optimizer_state,
                load_trainer_state=not cfg.reset_trainer_state,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
            log.info("Checkpoint successfully loaded")
            if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                trainer.scheduler = BoltOnWarmupScheduler.wrap(
                    trainer.scheduler,
                    trainer.global_step,
                    int(trainer.global_step + cfg.scheduler.t_warmup),
                )

        if cfg.force_save_unsharded and cfg.distributed_strategy != DistributedStrategy.ddp:
            log.info("Saving unsharded checkpoint...")
            checkpoint_path, _ = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")
    if torch.cuda.is_available():
        # Set CUDA device.
        torch.cuda.set_device(f"cuda:{get_local_rank()}")

        # Initialize process group.
        device_as_string = f"cuda:{get_local_rank()}"
        torch.cuda.set_device(
            device_as_string
        )  # Set this early to prevent GPU 0 from picking up a bunch of tensors it shouldn't have.
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=30), device_id=torch.device(device_as_string)
        )
    elif torch.backends.mps.is_available():
        if not os.getenv("RANK"):
            os.environ["RANK"] = "0"
        if not os.getenv("WORLD_SIZE"):
            os.environ["WORLD_SIZE"] = "1"
        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "0.0.0.0"
        if not os.getenv("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "24501"
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    else:
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise EvolutionCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    if torch.backends.mps.is_available():
        log.info("Device is MPS. Updating config...")
        cfg.model.init_device = "mps"
        cfg.distributed_strategy = "single"  # type: ignore

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log.info("Device is CPU. Updating config...")
        cfg.model.init_device = "cpu"
        cfg.distributed_strategy = "single"  # type: ignore
    main(cfg)
