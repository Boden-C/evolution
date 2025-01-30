from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..config import OptimizerConfig, SchedulerConfig


class Optimizer:
    def __init__(self, params: Iterable[object], cfg: OptimizerConfig):
        self.params = list(params)
        self.cfg = cfg
        self._optim = None
        try:
            import torch  # type: ignore

            if cfg.type.value == "adamw":
                self._optim = torch.optim.AdamW(self.params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
            elif cfg.type.value == "lionw":
                # Minimal Lion-like optimizer
                self._optim = _LionW(self.params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
            else:
                self._optim = torch.optim.Adam(self.params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        except Exception:
            self._optim = None

    def step(self) -> None:  # pragma: no cover - torch optional
        if self._optim is not None:
            self._optim.step()

    def zero_grad(self) -> None:  # pragma: no cover - torch optional
        if self._optim is not None:
            self._optim.zero_grad(set_to_none=True)

    def set_lr(self, lr: float) -> None:  # pragma: no cover - torch optional
        if self._optim is None:
            return
        for group in self._optim.param_groups:  # type: ignore[attr-defined]
            group["lr"] = lr

    def clip_grads_and_collect_metrics(self, max_norm: Optional[float] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if max_norm is None or max_norm <= 0:
            return metrics
        try:
            import torch  # type: ignore

            total_norm = torch.nn.utils.clip_grad_norm_(self.params, max_norm)
            metrics["grad_total_norm"] = float(getattr(total_norm, "item", lambda: total_norm)())
        except Exception:
            pass
        return metrics


@dataclass
class Scheduler:
    cfg: SchedulerConfig

    def __post_init__(self) -> None:
        self._step = 0

    def get_lr(self) -> float:
        warmup = max(0, int(self.cfg.warmup_steps))
        total = max(1, int(self.cfg.total_steps))
        s = self._step
        if self.cfg.type.value == "constant":
            return 1.0
        if s < warmup:
            return float(s + 1) / float(max(1, warmup))
        t = (s - warmup) / float(max(1, total - warmup))
        if self.cfg.type.value == "linear":
            return max(0.0, 1.0 - t)
        # cosine
        import math

        return 0.5 * (1.0 + math.cos(math.pi * t))

    def step(self) -> float:
        self._step += 1
        return self.get_lr()


def build_optimizer(params: Iterable[object], cfg: OptimizerConfig) -> Optimizer:
    return Optimizer(params, cfg)


def build_scheduler(cfg: SchedulerConfig) -> Scheduler:
    return Scheduler(cfg)


class _LionW:  # pragma: no cover - simple, local optimizer
    def __init__(self, params: Iterable[object], lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        try:
            import torch  # type: ignore

            self.param_groups: List[Dict] = [
                {"params": [p for p in params if getattr(p, "requires_grad", True)], "lr": lr, "betas": betas, "weight_decay": weight_decay}
            ]
            for g in self.param_groups:
                for p in g["params"]:
                    self.state = getattr(self, "state", {})
                    self.state[p] = {"exp_avg": torch.zeros_like(p.data)}
        except Exception:
            self.param_groups = []

    def step(self) -> None:
        try:
            import torch  # type: ignore

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    if wd != 0:
                        g = g.add(p.data, alpha=wd)
                    state = self.state[p]
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta2).add_(g, alpha=1 - beta2)
                    update = exp_avg.sign()
                    p.data.add_(update, alpha=-lr)
        except Exception:
            return

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in getattr(self, "param_groups", []):
            for p in group.get("params", []):
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        try:
                            p.grad.detach_()
                            p.grad.zero_()
                        except Exception:
                            p.grad = None


__all__ = ["Optimizer", "Scheduler", "build_optimizer", "build_scheduler"]
