from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainConfig
from ..model import Elevation
from ..torch_util import seed_all


@dataclass
class SpeedMonitor:
    window: int = 50


@dataclass
class LRMonitor:
    enabled: bool = True


@dataclass
class Trainer:
    cfg: TrainConfig
    speed: SpeedMonitor = SpeedMonitor()
    lrmon: LRMonitor = LRMonitor()

    def __post_init__(self) -> None:
        seed_all(self.cfg.seed)
        self.model = Elevation(self.cfg.model)

    def fit(self) -> None:  # pragma: no cover - placeholder
        return None

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - I/O shim
        return None


__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]
