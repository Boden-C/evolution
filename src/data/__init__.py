from __future__ import annotations

from typing import Any, Optional

from .collator import DataCollator, CustomDatasetDataCollator
from .iterable_dataset import ShardedIterableDataset
from .memmap_dataset import MemmapDataset
from ..config import TrainConfig


def build_collator(config: TrainConfig) -> DataCollator:
	return DataCollator.from_train_config(config)


def build_train_dataloader(
	dataset: Any,
	config: TrainConfig,
	*,
	batch_size: Optional[int] = None,
	shuffle: bool = True,
) -> Any:
	"""Build a DataLoader for training.

	Lazily imports torch and returns torch.utils.data.DataLoader.
	"""
	try:
		import torch  # noqa: F401
		from torch.utils.data import DataLoader
	except Exception as e:  # pragma: no cover - optional dependency
		raise RuntimeError(
			"PyTorch is required to build dataloaders. Please install torch."
		) from e

	collate_fn = build_collator(config)
	data_cfg = config.data
	bs = batch_size if batch_size is not None else config.per_device_batch_size

	return DataLoader(
		dataset,
		batch_size=bs,
		shuffle=shuffle,
		num_workers=data_cfg.num_workers,
		pin_memory=data_cfg.pin_memory,
		prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
		persistent_workers=data_cfg.persistent_workers if data_cfg.num_workers > 0 else False,
		collate_fn=collate_fn,
		timeout=data_cfg.timeout,
		drop_last=True,
	)


def build_eval_dataloader(
	dataset: Any,
	config: TrainConfig,
	*,
	batch_size: Optional[int] = None,
	shuffle: bool = False,
) -> Any:
	"""Build a DataLoader for evaluation.

	Lazily imports torch and returns torch.utils.data.DataLoader.
	"""
	try:
		import torch  # noqa: F401
		from torch.utils.data import DataLoader
	except Exception as e:  # pragma: no cover - optional dependency
		raise RuntimeError(
			"PyTorch is required to build dataloaders. Please install torch."
		) from e

	collate_fn = build_collator(config)
	data_cfg = config.data
	bs = batch_size if batch_size is not None else config.per_device_batch_size

	return DataLoader(
		dataset,
		batch_size=bs,
		shuffle=shuffle,
		num_workers=data_cfg.num_workers,
		pin_memory=data_cfg.pin_memory,
		prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
		persistent_workers=data_cfg.persistent_workers if data_cfg.num_workers > 0 else False,
		collate_fn=collate_fn,
		timeout=data_cfg.timeout,
		drop_last=False,
	)


__all__ = [
	"DataCollator",
	"CustomDatasetDataCollator",
	"build_collator",
	"build_train_dataloader",
	"build_eval_dataloader",
	"ShardedIterableDataset",
	"MemmapDataset",
]
