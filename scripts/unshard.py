import logging
import shutil
from pathlib import Path
from typing import Optional, Union


import torch

from src.core.checkpoint import build_sharded_checkpointer
from src.config import ShardedCheckpointerType, TrainConfig
from src.util.safetensors import state_dict_to_safetensors_file

logger = logging.getLogger(__name__)


def main(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sharded_checkpoint_type: Optional[ShardedCheckpointerType] = None,
    model_only: bool = False,
    safe_tensors: bool = False,
    use_shared_mem_impl: bool = False,
) -> None:
    """
    Unshard a sharded model checkpoint into a single checkpoint directory.
    Args:
        input_dir: Directory containing sharded checkpoint and config.yaml.
        output_dir: Directory to write unsharded checkpoint files.
        sharded_checkpoint_type: Type of sharded checkpoint (optional).
        model_only: If True, only unshard model weights.
        safe_tensors: If True, save weights in .safetensors format.
        use_shared_mem_impl: Use shared memory implementation for legacy checkpoints.
    """
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    config_path = input_dir / "config.yaml"
    if not config_path.is_file():
        logger.error("Missing config.yaml in input directory: %s", input_dir)
        raise FileNotFoundError(f"config.yaml not found in {input_dir}")
    config = TrainConfig.load(config_path, validate_paths=False)


    sharded_checkpoint_type = sharded_checkpoint_type or config.sharded_checkpointer
    checkpointer = build_sharded_checkpointer(
        config, name=sharded_checkpoint_type, use_shared_mem_impl=use_shared_mem_impl
    )


    model_state_dict, optim_state_dict, trainer_state_dict = checkpointer.unshard_checkpoint(
        input_dir,
        load_optimizer_state=not model_only,
        load_trainer_state=not model_only,
    )


    # Save model weights
    if safe_tensors:
        model_output = str(output_dir / "model.safetensors")
        logger.info("Saving model state to %s", model_output)
        state_dict_to_safetensors_file(model_state_dict, model_output)
    else:
        model_output = str(output_dir / "model.pt")
        logger.info("Saving model state to %s", model_output)
        torch.save(model_state_dict, model_output)
    del model_state_dict


    if not model_only:
        if optim_state_dict is None:
            logger.error("Optimizer state dict missing when model_only is False.")
            raise ValueError("Optimizer state dict is None.")

        # Save optimizer state
        if safe_tensors:
            optim_output = str(output_dir / "optim.safetensors")
            logger.info("Saving optimizer state to %s", optim_output)
            state_dict_to_safetensors_file(optim_state_dict, optim_output)
        else:
            optim_output = str(output_dir / "optim.pt")
            logger.info("Saving optimizer state to %s", optim_output)
            torch.save(optim_state_dict, optim_output)
        del optim_state_dict

        # Save trainer state
        train_output = str(output_dir / "train.pt")
        logger.info("Saving trainer state to %s", train_output)
        torch.save(trainer_state_dict, train_output)
        del trainer_state_dict


    # Copy config.yaml to output directory
    logger.info("Copying config.yaml to %s", output_dir)
    shutil.copy(config_path, output_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="unshard.py",
        description="Unshard a sharded model checkpoint into a single checkpoint directory."
    )
    parser.add_argument("input_dir", help="Directory containing sharded checkpoint and config.yaml.")
    parser.add_argument("output_dir", help="Directory to write unsharded checkpoint files.")
    parser.add_argument(
        "--type",
        choices=list(ShardedCheckpointerType),
        default=None,
        help="Type of sharded checkpoint. Defaults to value in config.yaml."
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only unshard model weights."
    )
    parser.add_argument(
        "--safe-tensors",
        action="store_true",
        help="Save weights in .safetensors format."
    )
    parser.add_argument(
        "--use-legacy-shared-mem-impl",
        action="store_true",
        help="Use shared memory implementation for legacy checkpoints."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        main(
            args.input_dir,
            args.output_dir,
            sharded_checkpoint_type=args.type,
            model_only=args.model_only,
            safe_tensors=args.safe_tensors,
            use_shared_mem_impl=args.use_legacy_shared_mem_impl,
        )
    except Exception as e:
        logger.error("Unshard failed: %s", e)
        exit(1)
