from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainConfig
from ..model import Elevation
from ..torch_util import seed_all
from .optimizer import build_optimizer, build_scheduler
from .checkpoint import Checkpointer
from ..text.tokenizer import Tokenizer


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
        self.tokenizer = Tokenizer(name_or_path=self.cfg.tokenizer.name_or_path, truncation_direction=self.cfg.tokenizer.truncation_direction)
        # Optimizer/scheduler/checkpointer
        try:
            import torch  # type: ignore

            self._params = [p for p in getattr(self.model, "_lm_head", [] ).parameters()]  # type: ignore[attr-defined]
            # Fallback: collect likely parameters from attributes
            if not self._params:
                self._params = []
                for attr in ("_embed", "_lm_head", "_pos", "_blocks"):
                    mod = getattr(self.model, attr, None)
                    if hasattr(mod, "parameters"):
                        self._params.extend(list(mod.parameters()))
        except Exception:
            self._params = []
        self.optimizer = build_optimizer(self._params, self.cfg.optimizer)
        self.scheduler = build_scheduler(self.cfg.scheduler)
        self.checkpointer = Checkpointer(self.cfg.save_folder)

    def fit(self) -> None:  # pragma: no cover - training skeleton
        # This is a stub demonstrating structure without data pipeline.
        # It shows LR schedule application and micro-batch loop wiring.
        try:
            import torch  # type: ignore

            total_steps = self.cfg.scheduler.total_steps
            for step in range(total_steps):
                lr_scale = self.scheduler.step()
                base_lr = self.cfg.optimizer.lr
                self.optimizer.set_lr(base_lr * lr_scale)
                # micro-batches would be processed here
                # self.optimizer.zero_grad(); loss.backward(); metrics = self.optimizer.clip_grads_and_collect_metrics(max_norm=1.0); self.optimizer.step()
                if (step + 1) % max(1, total_steps // 10) == 0:
                    self.checkpointer.save(name=f"step-{step+1}")
        except Exception:
            return None

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - I/O shim
        return None


__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]
