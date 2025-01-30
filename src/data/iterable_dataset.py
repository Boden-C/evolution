from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence, Tuple, TypeVar
import random

T = TypeVar("T")


class IterableDataset:
    def __init__(self, data: Iterable[T]):
        self._data = data

    def __iter__(self) -> Iterator[T]:  # pragma: no cover - simple
        return iter(self._data)


class ShardedIterableDataset(IterableDataset):
    """Shard an iterable dataset across ranks.

    Parameters:
        data: The underlying iterable or sequence.
        rank: Current process rank in [0, world_size).
        world_size: Total number of processes.
        seed: Optional base seed to shuffle per epoch.
    """

    def __init__(
        self,
        data: Iterable[T] | Sequence[T],
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: Optional[int] = None,
        epoch: int = 0,
    ):
        super().__init__(data)
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if not (0 <= rank < world_size):
            raise ValueError("rank must be in [0, world_size)")
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch
        self._is_sequence = isinstance(data, Sequence)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _indices(self, n: int) -> Iterator[int]:
        idxs = list(range(n))
        if self.seed is not None:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(idxs)
        # round-robin shard
        return iter(idxs[self.rank :: self.world_size])

    def __iter__(self) -> Iterator[T]:  # pragma: no cover - simple
        if self._is_sequence:
            data_seq = self._data  # type: ignore[assignment]
            assert isinstance(data_seq, Sequence)
            for i in self._indices(len(data_seq)):
                yield data_seq[i]
        else:
            # Materialize to list only once per epoch for consistent sharding.
            items = list(self._data)
            for i in self._indices(len(items)):
                yield items[i]


__all__ = ["IterableDataset", "ShardedIterableDataset"]
