from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from ..config import OptimizerConfig, SchedulerConfig
from ..logger import get_logger

if TYPE_CHECKING:
    from torch import Tensor  # noqa: F401

try:
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
    from torch.optim import Optimizer as TorchOptimizer  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    dist = None  # type: ignore
    TorchOptimizer = object  # type: ignore


log = get_logger(__name__)


# ------------------------------
# Utilities
# ------------------------------

def _default_device() -> "torch.device":  # type: ignore[return-type]
    if torch is None:  # pragma: no cover - torch optional
        return None  # type: ignore
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_distributed() -> bool:
    if dist is None:
        return False
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


# ------------------------------
# Optimizers
# ------------------------------

class BaseOptimizer(TorchOptimizer):  # type: ignore[misc]
    """Enterprise-ready optimizer base with optional metric collection and clipping.

    Public API kept compatible with previous wrapper: step(), zero_grad(), set_lr(),
    clip_grads_and_collect_metrics(max_norm).
    """

    def __init__(self, *args: Any, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._record_update_metrics = bool(record_update_metrics)
        self._collecting_metrics = False
        self._selective_updates = bool(selective_updates)

    def set_lr(self, lr: float) -> None:  # pragma: no cover - torch optional
        if not hasattr(self, "param_groups"):
            return
        for group in self.param_groups:  # type: ignore[attr-defined]
            group["lr"] = lr

    def zero_grad(self) -> None:  # pragma: no cover - torch optional
        if hasattr(super(), "zero_grad"):
            try:
                super().zero_grad(set_to_none=True)  # type: ignore[misc]
                return
            except TypeError:
                pass
        # Fallback
        for group in getattr(self, "param_groups", []):  # type: ignore[attr-defined]
            for p in group.get("params", []):
                if getattr(p, "grad", None) is not None:
                    try:
                        p.grad.detach_()
                        p.grad.zero_()
                    except Exception:
                        p.grad = None

    def get_state_for_param(self, param: Any) -> Dict[str, Optional[Tensor]]:  # type: ignore[name-defined]
        del param
        return {}

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def clip_grads_and_collect_metrics(self, max_norm: Optional[float] = None) -> Dict[str, float]:
        """Clip gradients globally and return light-weight metrics.

        - max_norm: if provided and > 0, performs global norm clipping.
        Returns dict with at least grad_total_norm. Compatible with previous API.
        """
        metrics: Dict[str, float] = {}
        if torch is None:
            return metrics

        self._collecting_metrics = True
        # Collect gradients
        grads: List[Tensor] = []  # type: ignore[name-defined]
        for group in self.param_groups:  # type: ignore[attr-defined]
            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is not None:
                    grads.append(g)
        if not grads:
            self._collecting_metrics = False
            return metrics

        total_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.detach(), 2.0, dtype=torch.float32) for g in grads], dim=0),
            2.0,
            dtype=torch.float32,
        )
        metrics["grad_total_norm"] = float(total_norm.item())

        if max_norm is not None and max_norm > 0:
            try:
                clip_coef = float(max_norm) / float(total_norm.item() + 1e-6)
                clip_coef = min(1.0, clip_coef)
                if clip_coef < 1.0:
                    for g in grads:
                        g.detach().mul_(clip_coef)
                    metrics["clipping_rate"] = 1.0
                else:
                    metrics["clipping_rate"] = 0.0
            except Exception:
                pass
        self._collecting_metrics = False
        return metrics


class LionW(BaseOptimizer):  # pragma: no cover - simple math
    """LionW optimizer with optional selective updates and update-cosine metric.

    Uses sign(EMA(beta1)* + grad*(1-beta1)) update and decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        record_update_metrics: bool = False,
        selective_updates: bool = False,
    ) -> None:
        assert lr > 0.0
        assert 0.0 <= betas[0] <= 1.0 and 0.0 <= betas[1] <= 1.0
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults, record_update_metrics=record_update_metrics, selective_updates=selective_updates)
        for g in self.param_groups:  # type: ignore[attr-defined]
            g["initial_lr"] = g["lr"]
        self._upd_dot: Optional[Tensor] = None  # type: ignore[name-defined]
        self._upd_norm: Optional[Tensor] = None  # type: ignore[name-defined]
        self._sgn_norm: Optional[Tensor] = None  # type: ignore[name-defined]

    def step(self, closure: Optional[Any] = None) -> None:  # type: ignore[override]
        if torch is None:
            return
        if closure is not None:
            with torch.enable_grad():
                closure()

        track = self._collecting_metrics and self._record_update_metrics
        upd_dot = torch.tensor(0.0, dtype=torch.float32) if track else None
        upd_norms: List[Tensor] = [] if track else None  # type: ignore[assignment]
        sgn_norms: List[Tensor] = [] if track else None  # type: ignore[assignment]

        for group in self.param_groups:  # type: ignore[attr-defined]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue

                # Decoupled weight decay (selective if enabled)
                if self._selective_updates:
                    mask = (grad != 0).to(p)  # type: ignore[attr-defined]
                    p.data.mul_(1 - mask * (lr * wd))
                else:
                    if wd:
                        p.data.mul_(1 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]

                # Compute update and apply
                update = m.mul(beta1).add(grad, alpha=1 - beta1)
                if self._selective_updates:
                    update = update.mul((grad != 0).to(update))
                signed = update.sign()
                p.data.add_(signed, alpha=-lr)

                # Momentum ema update
                m.mul_(beta2).add(grad, alpha=1 - beta2)

                if track:
                    upd_dot = upd_dot.to(update.device)
                    upd_dot += torch.tensordot(update, signed, dims=len(update.shape))
                    upd_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))  # type: ignore[union-attr]
                    sgn_norms.append(torch.linalg.vector_norm(signed, 2.0, dtype=torch.float32))  # type: ignore[union-attr]

        if track and upd_dot is not None and upd_norms is not None and sgn_norms is not None:
            self._upd_dot = upd_dot.to(_default_device())
            self._upd_norm = torch.linalg.vector_norm(torch.stack(upd_norms), 2.0, dtype=torch.float32).to(_default_device())
            self._sgn_norm = torch.linalg.vector_norm(torch.stack(sgn_norms), 2.0, dtype=torch.float32).to(_default_device())

    def get_post_step_metrics(self) -> Dict[str, Tensor]:  # type: ignore[name-defined]
        if torch is None:
            return {}
        if self._upd_dot is None or self._upd_norm is None or self._sgn_norm is None:
            return {}
        denom = torch.clamp(self._upd_norm * self._sgn_norm, min=1e-8)
        cos_sim = self._upd_dot / denom
        self._upd_dot = None
        self._upd_norm = None
        self._sgn_norm = None
        return {"update_cos_sim": cos_sim}


class AdamW(BaseOptimizer, torch.optim.AdamW if torch is not None else object):  # type: ignore[misc]
    """AdamW with optional selective updates and step-size metrics via param diff.

    When metrics are enabled, captures per-param update norms and max magnitude by
    diffing parameters around the super().step() call to avoid duplicating PyTorch logic.
    """

    def __init__(self, *args: Any, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs: Any):
        super().__init__(
            *args,
            record_update_metrics=record_update_metrics,
            selective_updates=selective_updates,
            **kwargs,
        )
        self._step_param_names: Optional[List[str]] = None
        self._step_update_norms: Optional[List[Tensor]] = None  # type: ignore[name-defined]
        self._step_update_maxs: Optional[List[Tensor]] = None  # type: ignore[name-defined]

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def step(self, closure: Optional[Any] = None) -> None:  # type: ignore[override]
        if torch is None:
            return
        # Selective decoupled weight decay before step (match AdamW behavior)
        if self._selective_updates:
            for g in self.param_groups:  # type: ignore[attr-defined]
                lr = g["lr"]
                wd = g.get("weight_decay", 0.0)
                if not wd:
                    continue
                for p in g["params"]:
                    grad = getattr(p, "grad", None)
                    if grad is None:
                        continue
                    mask = (grad != 0).to(p)
                    p.data.mul_(1 - mask * (lr * wd))
                # Reset weight_decay for super().step to avoid double application
                g["weight_decay"] = 0.0

        # Capture parameter snapshots only when we need metrics
        capture = self._record_update_metrics and self._collecting_metrics
        pre_params: Optional[List[Tensor]] = [] if capture else None  # type: ignore[name-defined]
        if capture:
            for g in self.param_groups:  # type: ignore[attr-defined]
                for p in g["params"]:
                    pre_params.append(p.data.detach().clone())  # type: ignore[union-attr]

        # Delegate to PyTorch AdamW
        if hasattr(super(), "step"):
            super().step(closure=closure)  # type: ignore[misc]

        # Restore configured weight decay if we changed it for selective updates
        if self._selective_updates:
            for g in self.param_groups:  # type: ignore[attr-defined]
                if "configured_weight_decay" in g:
                    g["weight_decay"] = g["configured_weight_decay"]
                # else leave at 0.0 which is effectively configured

        if not capture:
            return

        device = _default_device()
        names: List[str] = []
        norms: List[Tensor] = []  # type: ignore[name-defined]
        maxs: List[Tensor] = []  # type: ignore[name-defined]
        idx = 0
        for g in self.param_groups:  # type: ignore[attr-defined]
            group_names = g.get("param_names", [""] * len(g.get("params", [])))
            for name, p in zip(group_names, g["params"]):
                names.append(str(name))
                before = pre_params[idx]
                after = p.data
                delta = (after - before).detach()
                norms.append(torch.linalg.vector_norm(delta, 2.0, dtype=torch.float32).unsqueeze(0))
                maxs.append(delta.abs().max().unsqueeze(0))
                idx += 1

        self._step_param_names = names
        self._step_update_norms = norms
        self._step_update_maxs = maxs

    def get_state_for_param(self, param: Any) -> Dict[str, Optional[Tensor]]:  # type: ignore[name-defined]
        if torch is None:
            return {}
        state = self.state.get(param, {})
        return {k: state.get(k) for k in ("exp_avg", "exp_avg_sq")}  # type: ignore[return-value]

    def get_post_step_metrics(self) -> Dict[str, Tensor]:  # type: ignore[name-defined]
        if torch is None:
            return {}
        names = self._step_param_names
        norms = self._step_update_norms
        maxs = self._step_update_maxs
        if names is None or norms is None or maxs is None:
            return {}
        metrics: Dict[str, Tensor] = {}
        for n, sn, sm in zip(names, norms, maxs):
            metrics[f"step/{n}.norm"] = sn.squeeze(0)
            metrics[f"step/{n}.max"] = sm.squeeze(0)
        self._step_param_names = None
        self._step_update_norms = None
        self._step_update_maxs = None
        return metrics


# ------------------------------
# Schedulers (multiplier semantics preserved)
# ------------------------------

@dataclass
class Scheduler:
    """Learning-rate multiplier scheduler with warmup and decay variants.

    get_lr() returns a multiplier to apply to the base LR, preserving existing behavior.
    """

    type: Any  # keep compatible with existing config enums/values
    warmup_steps: int = 0
    total_steps: int = 1
    alpha: float = 0.0  # final multiplier at the end for decay schedulers

    def __post_init__(self) -> None:
        self._step = 0

    def _linear_warmup_mul(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        s = min(step + 1, self.warmup_steps)
        return float(s) / float(max(1, self.warmup_steps))

    def _cosine_decay_mul(self, step: int) -> float:
        import math
        t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        t = max(0.0, min(1.0, t))
        return self.alpha + (1.0 - self.alpha) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _linear_decay_mul(self, step: int) -> float:
        t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        t = max(0.0, min(1.0, t))
        return max(self.alpha, 1.0 - (1.0 - self.alpha) * t)

    def get_lr(self) -> float:
        """Return LR multiplier for the current step."""
        name = getattr(self.type, "value", self.type)
        s = self._step
        if name in ("constant", "const"):
            if self.warmup_steps > 0 and s < self.warmup_steps:
                return self._linear_warmup_mul(s)
            return 1.0
        if s < self.warmup_steps:
            return self._linear_warmup_mul(s)
        if name in ("linear", "linear_with_warmup"):
            return self._linear_decay_mul(s)
        # default to cosine family
        return self._cosine_decay_mul(s)

    def step(self) -> float:
        self._step += 1
        return self.get_lr()


# ------------------------------
# Builders
# ------------------------------

def build_optimizer(params: Iterable[Any], cfg: OptimizerConfig) -> BaseOptimizer:
    """Factory for project optimizers.

    - Supports AdamW and LionW.
    - Keeps torch optional; returns a no-op stub if torch is missing.
    """
    name = getattr(cfg.type, "value", cfg.type)
    if torch is None:  # pragma: no cover - torch optional
        log.warning("torch not available; returning no-op optimizer")
        # Minimal stub matching API
        class _NoOp(BaseOptimizer):  # type: ignore
            def __init__(self) -> None:  # type: ignore
                super().__init__([{}], {})  # type: ignore
            def step(self, closure: Optional[Any] = None) -> None:  # noqa: D401
                return
        return _NoOp()

    if str(name).lower() == "lionw":
        return LionW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if str(name).lower() == "adamw":
        return AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    # Fallback to Adam
    return AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)


def build_scheduler(cfg: SchedulerConfig) -> Scheduler:
    """Factory for project schedulers.

    Preserves legacy semantics: returns LR multiplier scheduler.
    """
    name = getattr(cfg.type, "value", cfg.type)
    warmup = int(getattr(cfg, "warmup_steps", 0) or 0)
    total = int(getattr(cfg, "total_steps", 1) or 1)
    # alpha is the final multiplier for decay schedulers (defaults to 0.0)
    alpha = float(getattr(cfg, "alpha_f", 0.0) or 0.0)
    if str(name).lower() in ("constant", "const"):
        return Scheduler(type="constant", warmup_steps=warmup, total_steps=total)
    if str(name).lower() in ("linear", "linear_with_warmup"):
        return Scheduler(type="linear", warmup_steps=warmup, total_steps=total, alpha=alpha if alpha > 0 else 0.0)
    # default to cosine with warmup
    return Scheduler(type="cosine", warmup_steps=warmup, total_steps=total, alpha=alpha if alpha > 0 else 0.0)


__all__ = [
    "BaseOptimizer",
    "LionW",
    "AdamW",
    "Scheduler",
    "build_optimizer",
    "build_scheduler",
]
