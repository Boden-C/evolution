"""
Script to display the total and non-embedding parameter count for a model defined by a training configuration file.

Usage:
    python scripts/show_model_size.py <CONFIG_PATH> [OPTIONS]

Arguments:
    CONFIG_PATH: Path to the YAML configuration file for model training.
    OPTIONS: Optional CLI overrides for config values.

This script loads a model configuration, instantiates a single-layer model, and estimates the total and non-embedding parameter count for the full model.
"""

import logging
import sys


from src.model import Model
from src.config import TrainConfig
from src.exceptions import ModelCliError
from src.util import clean_opt, prepare_cli_environment


def setup_logging():
    """Configure logging for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

log = setup_logging()




def main(cfg: TrainConfig) -> None:
    """
    Calculates and logs the total and non-embedding parameter count for the model.

    Args:
        cfg (TrainConfig): Training configuration object.
    """
    # Always run on CPU for parameter counting
    cfg.model.init_device = "cpu"
    n_layers = getattr(cfg.model, "n_layers", 1)
    cfg.model.n_layers = 1

    # Instantiate a single-layer model to estimate per-layer parameters
    try:
        single_layer_model = Model(cfg.model)
        block = getattr(single_layer_model.transformer, "blocks", [None])[0]
        if block is None:
            raise AttributeError("No blocks found in transformer.")
        params_per_block = sum(p.numel() for p in block.parameters())
    except Exception as exc:
        log.error(f"Model structure does not match expected format: {exc}")
        return

    # Estimate total parameters for full model
    total_params = single_layer_model.num_params() + (params_per_block * (n_layers - 1))
    non_embedding_params = single_layer_model.num_params(include_embedding=False) + (params_per_block * (n_layers - 1))

    log.info(f"Total number of parameters: {total_params:,d}")
    log.info(f"Number of non-embedding parameters: {non_embedding_params:,d}")


if __name__ == "__main__":
    prepare_cli_environment()
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <CONFIG_PATH> [OPTIONS]", file=sys.stderr)
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    # Add minimal required overrides to avoid unnecessary data loading or saving
    cli_overrides = args_list + ["--data.paths=[]", "--save_folder=/tmp", "--evaluators=[]"]
    try:
        cfg = TrainConfig.load(
            yaml_path,
            [clean_opt(s) for s in cli_overrides],
            validate_paths=False,
        )
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        sys.exit(2)
    main(cfg)
