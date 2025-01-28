from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import OptimizerConfig, SchedulerConfig


class Optimizer:
    def __init__(self, params: Iterable[object], cfg: OptimizerConfig):
        self.params = list(params)
        self.cfg = cfg

    def step(self) -> None:  # pragma: no cover - placeholder
        return None

    def zero_grad(self) -> None:  # pragma: no cover - placeholder
        return None


@dataclass
class Scheduler:
    cfg: SchedulerConfig

    def step(self) -> None:  # pragma: no cover - placeholder
        return None


def build_optimizer(params: Iterable[object], cfg: OptimizerConfig) -> Optimizer:
    return Optimizer(params, cfg)


def build_scheduler(cfg: SchedulerConfig) -> Scheduler:
    return Scheduler(cfg)


__all__ = ["Optimizer", "Scheduler", "build_optimizer", "build_scheduler"]
