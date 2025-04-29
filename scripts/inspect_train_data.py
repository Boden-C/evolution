
"""
Inspect training data batches for a given run.
This script provides utilities to inspect and debug training data batches for a specific model run.
Supports inspection with or without device data indices, and handles custom mount points.
"""

import argparse
import gzip
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from src.util.path import cached_path
from src.util.checkpoint import load_state_dict
from src.config import TrainConfig
from src.data import build_memmap_dataset, build_train_dataloader
from src.data.iterable_dataset import IterableDataset
from src.text.tokenizer import Tokenizer
from src.util.env import add_cached_path_clients, prepare_cli_environment
from src.util.cli import clean_opt



def get_global_train_examples_seen_before_step(step: int, trainer_state: dict, cfg: TrainConfig) -> int:
    """
    Returns the number of global training examples seen before a given step.
    Raises ValueError if the requested step is not after the current global step.
    Args:
        step: Target training step.
        trainer_state: Trainer state dictionary.
        cfg: Training configuration.
    Returns:
        Number of global training examples seen before the given step.
    """
    global_step = trainer_state["global_step"]
    if global_step >= step:
        raise ValueError(f"Step {step} must be after training first step {global_step}")
    # Prefer most specific key, fallback to older keys for compatibility
    global_train_examples_seen_this_epoch = trainer_state.get(
        "global_train_examples_seen_this_epoch",
        trainer_state.get(
            "global_train_examples_seen",
            trainer_state.get("global_data_step", global_step) * cfg.global_train_batch_size,
        ),
    )
    global_train_examples_seen_this_epoch += (step - 1 - global_step) * cfg.global_train_batch_size
    return global_train_examples_seen_this_epoch



def _revert_data_mounts(cfg: TrainConfig, mounts: List[Tuple[str, str]]):
    """
    Revert data paths in config to their original mount sources.
    Args:
        cfg: Training configuration.
        mounts: List of (source, target) mount tuples.
    """
    if not getattr(cfg.data, "paths", None):
        return
    new_paths = []
    for path in cfg.data.paths:
        new_path = path
        for source, target in mounts:
            if path.startswith(target):
                new_path = source + path[len(target):]
                break
        new_paths.append(new_path)
    cfg.data.paths = new_paths



def inspect_data_without_device_data_indices(
    run_path: str,
    *steps: int,
    world_size: int,
    ranks: List[int],
    reference_step: int,
    mounts: Optional[List[Tuple[str, str]]] = None,
):
    """
    Inspect training data batches for a run when device data indices are not available.
    Args:
        run_path: Path to the run directory.
        steps: Training steps to inspect.
        world_size: Number of distributed devices.
        ranks: List of device ranks to inspect.
        reference_step: Step number for checkpoint reference.
        mounts: Optional list of mount tuples.
    """
    cfg = TrainConfig.load(
        cached_path(os.path.join(run_path, f"step{reference_step}/config.yaml")),
        overrides=[clean_opt("--evaluators=[]"), clean_opt("--save_overwrite")],
    )
    cfg.data.num_workers = 1
    if mounts:
        _revert_data_mounts(cfg, mounts)
    if cfg.global_train_batch_size % world_size != 0:
        raise ValueError(f"World size must divide global_train_batch_size {cfg.global_train_batch_size}")
    cfg.device_train_batch_size = cfg.global_train_batch_size // world_size
    trainer_state = None
    for checkpoint_path in [
        f"step{reference_step}/train/rank0.pt",
        f"step{reference_step}/train.pt",
        f"step{reference_step}/rank0.pt"
    ]:
        try:
            trainer_state = load_state_dict(run_path, checkpoint_path, map_location="cpu")
            break
        except FileNotFoundError:
            continue
    if trainer_state is None:
        raise FileNotFoundError("No valid trainer state checkpoint found.")
    tokenizer = Tokenizer.from_train_config(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_folder = tmpdir
        os.environ["RANK"] = "0"
        dataloader = build_train_dataloader(cfg, world_size=world_size)
        for rank in ranks:
            os.environ["RANK"] = str(rank)
            os.environ["FS_LOCAL_RANK"] = "1"
            for step in steps:
                dataloader = build_train_dataloader(cfg, world_size=world_size)
                if not isinstance(dataloader.dataset, IterableDataset):
                    raise TypeError("Dataloader dataset must be IterableDataset.")
                dataloader.dataset.start_index = get_global_train_examples_seen_before_step(
                    step, trainer_state, cfg
                )
                batch = next(iter(dataloader))
                for i, (batch_entry, instance_mask) in enumerate(
                    zip(batch["input_ids"].tolist(), batch["instance_mask"].tolist())
                ):
                    masked_instance = not instance_mask
                    example = tokenizer.decode(batch_entry)
                    print(f'[step={step}, rank={rank}, example={i}, masked={masked_instance}] "{example}"\n')



def main(
    run_path: str,
    *steps: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    reference_step: Optional[int] = None,
    use_data_indices: bool = True,
    mounts: Optional[List[Tuple[str, str]]] = None,
):
    """
    Main entry point for inspecting training data batches for a given run.
    Handles both cases: with and without device data indices.
    Args:
        run_path: Path to run directory.
        steps: Training steps to inspect.
        world_size: Number of distributed devices.
        rank: Device rank to inspect (None for all).
        reference_step: Step number for checkpoint reference.
        use_data_indices: Whether to use device data indices if present.
        mounts: Optional list of mount tuples.
    """
    save_folder = Path(run_path)
    if not use_data_indices or not (save_folder / "data-indices").is_dir():
        if world_size is None or reference_step is None:
            raise ValueError("world_size and reference_step required when data indices are not present.")
        ranks = [rank] if rank is not None else list(range(world_size))
        inspect_data_without_device_data_indices(
            run_path,
            *steps,
            world_size=world_size,
            ranks=ranks,
            reference_step=reference_step,
            mounts=mounts,
        )
        return
    cfg = TrainConfig.load(save_folder / "config.yaml", overrides=[clean_opt("--evaluators=[]")])
    dataset = build_memmap_dataset(cfg, cfg.data)
    tokenizer = Tokenizer.from_train_config(cfg)
    if rank is None:
        num_indices_files = len(list((save_folder / "data-indices").glob("*.tsv.gz")))
        if world_size is not None and world_size != num_indices_files:
            raise ValueError(f"World size {world_size} does not match number of indices files {num_indices_files}")
        indices_files = {
            r: gzip.open(save_folder / "data-indices" / f"rank{r}.tsv.gz", "rt")
            for r in range(num_indices_files)
        }
    else:
        indices_files = {rank: gzip.open(save_folder / "data-indices" / f"rank{rank}.tsv.gz", "rt")}
    try:
        for step in sorted(steps):
            for r in sorted(indices_files.keys()):
                for line in indices_files[r]:
                    if line.startswith(f"{step}\t"):
                        indices = [int(i) for i in line.strip().split("\t")[1:]]
                        for i, index in enumerate(indices):
                            token_ids = dataset[index]["input_ids"]
                            example = tokenizer.decode(token_ids.tolist())
                            print(f'[step={step}, rank={r}, example={i}] "{example}"\n')
    finally:
        for f in indices_files.values():
            f.close()



if __name__ == "__main__":
    prepare_cli_environment()
    add_cached_path_clients()
    parser = argparse.ArgumentParser(
        description="Inspect training data batches for a given run."
    )
    parser.add_argument("run_path", help="Path to run for training data inspection.")
    parser.add_argument(
        "rank",
        type=int,
        help="Device rank to inspect. Use -1 for all ranks.",
    )
    parser.add_argument("steps", nargs="+", type=int, help="Steps to inspect.")
    parser.add_argument(
        "--no_data_indices",
        action="store_false",
        dest="use_data_indices",
        help="Ignore data indices if present.",
    )
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        help="Checkpoint step number for training state. Required if no data indices.",
    )
    parser.add_argument("--world_size", type=int, help="World size. Required if no data indices.")
    parser.add_argument(
        "--mount",
        default=[],
        nargs=2,
        action="append",
        dest="mounts",
        help="Directory mounts used in original run. Example: weka:// /weka/",
    )
    args = parser.parse_args()
    main(
        args.run_path,
        *args.steps,
        world_size=args.world_size,
        rank=args.rank if args.rank >= 0 else None,
        reference_step=args.checkpoint_num if args.checkpoint_num is not None and args.checkpoint_num >= 0 else None,
        use_data_indices=args.use_data_indices,
        mounts=args.mounts,
    )
