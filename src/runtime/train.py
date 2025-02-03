from __future__ import annotations

import functools
import gc
from ..logger import get_logger
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union, TextIO

# Internal imports
from ..config import TrainConfig
from ..model import Elevation
from ..torch_util import seed_all
from .optimizer import build_optimizer, build_scheduler
from .checkpoint import Checkpointer
from ..text.tokenizer import Tokenizer

# Optional heavy deps guarded to keep this file import-safe and not runnable by default
try:  # pragma: no cover - soft dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    np = None  # type: ignore

try:  # pragma: no cover - soft dependency
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    torch = None  # type: ignore
    F = None  # type: ignore

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import wandb
import cProfile
from pstats import SortKey

log = get_logger(__name__)


# ----------------------------
# Monitors
# ----------------------------
@dataclass
class SpeedMonitor:
    window: int = 50
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0.0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        *,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int = 0,
        num_bck_flops: int = 0,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        total_flops = (num_fwd_flops + num_bck_flops) * max(global_total_tokens, 1)
        self.total_training_Gflops = float(total_flops) / 1e9 if total_flops else 0.0
        if record:
            if len(self.start_times) >= self.window:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(int(device_batch_num_tokens))

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "throughput/total_tokens": float(self.global_total_tokens),
            "throughput/total_training_Gflops": float(self.total_training_Gflops),
        }
        if self.total_training_Gflops > 0:
            try:
                metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)
            except ValueError:
                pass
        if self.start_times:
            interval_seconds = max(time.monotonic() - self.start_times[0], 1e-9)
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = float(interval_tokens) / interval_seconds
            metrics["throughput/device/batches_per_second"] = float(interval_batches) / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    enabled: bool = True
    optim: Any = None

    def check(self) -> Dict[str, float]:
        if not self.enabled or self.optim is None:
            return {}
        try:
            groups = getattr(self.optim, "param_groups", [])
            return {f"optim/learning_rate_group{idx}": float(g.get("lr", 0.0)) for idx, g in enumerate(groups)}
        except Exception:
            return {}


# ----------------------------
# Losses
# ----------------------------

def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    if torch is None or F is None:  # pragma: no cover - soft dependency guard
        raise RuntimeError("torch is required for loss computation but is not available")
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if not compute_z_loss:
        return loss, None
    # Auxiliary z-loss on log-sum-exp of logits to stabilize softmax
    z_squared = logits.logsumexp(-1).pow(2)
    mask = (labels != ignore_index).to(z_squared.dtype)
    if reduction == "mean":
        z_squared = (z_squared * mask).mean()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()
    z_loss = z_loss_multiplier * z_squared
    return loss, z_loss


fused_loss_fn: Optional[Callable]
try:  # pragma: no cover - optional fused loss
    import flash_attn  # type: ignore
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as flash_cross_entropy_loss  # type: ignore

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # flash-attn API renamed parameter from ignored_index -> ignore_index in v2.5.8
        from packaging import version as _v  # type: ignore

        ce_loss_use_ignore_index_param = _v.parse(flash_attn.__version__) >= _v.parse("2.5.8")
        ignore_index_kwarg = {("ignore_index" if ce_loss_use_ignore_index_param else "ignored_index"): ignore_index}
        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )
        mask = (labels != ignore_index)
        if reduction == "mean":
            denom = mask.sum().clamp_min(1)
            loss = loss.sum() / denom
            if compute_z_loss and z_loss is not None:
                z_loss = z_loss.sum() / denom
        elif reduction == "sum":
            loss = loss.sum()
            if compute_z_loss and z_loss is not None:
                z_loss = z_loss.sum()
        return loss, (z_loss if compute_z_loss else None)
except Exception:  # pragma: no cover - if flash-attn not present
    fused_loss_fn = None


# ----------------------------
# Trainer
# ----------------------------
@dataclass
class Trainer:
    cfg: TrainConfig
    speed: SpeedMonitor = SpeedMonitor()
    lrmon: LRMonitor = LRMonitor()

    # Runtime-managed fields
    model: Optional[Elevation] = field(default=None, init=False)
    tokenizer: Optional[Tokenizer] = field(default=None, init=False)
    optimizer: Any = field(default=None, init=False)
    scheduler: Any = field(default=None, init=False)
    checkpointer: Optional[Checkpointer] = field(default=None, init=False)
    dist_model: Union[DDP, FSDP] = field(default=None, init=False)

    # State
    epoch: int = field(default=0, init=False)
    global_step: int = field(default=0, init=False)
    global_train_examples_seen_this_epoch: int = field(default=0, init=False)
    global_train_tokens_seen: int = field(default=0, init=False)
    min_train_loss: float = field(default=float("inf"), init=False)
    cur_train_loss: float = field(default=float("inf"), init=False)

    # Options
    loss_fn: Callable[..., Any] = field(default=cross_entropy_loss, init=False)
    device: Optional["torch.device"] = field(default=None, init=False)  # type: ignore[name-defined]

    # Evaluation
    evaluators: List[Any] = field(default_factory=list)
    indices_file: Optional[TextIO] = None

    # Timing and state
    _start_time: float = 0.0
    _gc_init_state: bool = True

    def __post_init__(self) -> None:
        # Seeding
        seed_all(self.cfg.seed)

        # Model + tokenizer
        self.model = Elevation(self.cfg.model)
        self.tokenizer = Tokenizer(
            name_or_path=self.cfg.tokenizer.name_or_path,
            truncation_direction=self.cfg.tokenizer.truncation_direction,
        )

        # Optional fused loss toggle
        try:
            if bool(getattr(self.cfg, "fused_loss", False)):
                if fused_loss_fn is None:
                    raise NameError("fused loss requested but flash-attn not available")
                self.loss_fn = fused_loss_fn  # type: ignore[assignment]
        except Exception:
            pass

        # Optimizer/scheduler/checkpointer setup
        self._params: List[Any] = []
        try:
            # common submodules to collect parameters from
            for attr in ("_embed", "_lm_head", "_pos", "_blocks"):
                mod = getattr(self.model, attr, None)
                if hasattr(mod, "parameters"):
                    self._params.extend(list(mod.parameters()))  # type: ignore[attr-defined]
            if not self._params and hasattr(self.model, "parameters"):
                self._params = list(self.model.parameters())  # type: ignore[attr-defined]
        except Exception:
            self._params = []

        self.optimizer = build_optimizer(self._params, self.cfg.optimizer)
        self.scheduler = build_scheduler(self.cfg.scheduler)
        self.checkpointer = Checkpointer(self.cfg.save_folder)

        # Device best-effort detection
        if torch is not None:  # pragma: no cover - soft dependency
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            try:
                self.model.to(self.device)  # type: ignore[union-attr]
            except Exception:
                pass

        # LR monitor can read from our optimizer
        self.lrmon.optim = self.optimizer

        # setup dist_model
        self.dist_model = DDP(self.model) if self.device and torch.cuda.device_count()>1 else self.model

        # setup wandb
        if self.cfg.wandb is not None:
            wandb.init(**self.cfg.wandb)

    # ----------------------------
    # Duration helpers
    # ----------------------------
    @property
    def tokens_per_batch(self) -> int:
        seq_len = int(getattr(self.cfg.model, "max_sequence_length", 0) or 0)
        gbs = int(getattr(self.cfg, "global_train_batch_size", 0) or 0)
        return gbs * seq_len

    @property
    def max_steps(self) -> int:
        md = getattr(self.cfg, "max_duration", None)
        if isinstance(md, int):
            return int(md)
        if isinstance(md, str):
            val = md.strip()
            if val.endswith("T"):
                # token-based duration
                max_tokens = int(float(val[:-1]))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                return self.global_step + math.ceil(tokens_remaining / max(self.tokens_per_batch, 1))
            if val.endswith("ep"):
                # epoch-based duration, approximate if batches_per_epoch known
                try:
                    bpe = int(getattr(self.dataset, "total_size", 0)) // int(
                        getattr(self.cfg, "global_train_batch_size", 1)
                    )
                except Exception:
                    bpe = 0
                max_epochs = int(float(val[:-2]))
                return max_epochs * bpe if bpe > 0 else int(max_epochs)
            return int(float(val))
        # default safe fallback
        return int(getattr(self.cfg.scheduler, "total_steps", 0) or 0)

    @property
    def max_tokens(self) -> int:
        md = getattr(self.cfg, "max_duration", None)
        if isinstance(md, int):
            return self.global_train_tokens_seen + max(md - self.global_step, 0) * max(self.tokens_per_batch, 1)
        if isinstance(md, str):
            val = md.strip()
            if val.endswith("T"):
                return int(float(val[:-1]))
            if val.endswith("ep"):
                try:
                    bpe = int(getattr(self.dataset, "total_size", 0)) // int(
                        getattr(self.cfg, "global_train_batch_size", 1)
                    )
                except Exception:
                    bpe = 0
                return int(float(val[:-2])) * bpe * max(self.tokens_per_batch, 1) if bpe > 0 else 0
            return self.global_train_tokens_seen + max(int(float(val)) - self.global_step, 0) * max(
                self.tokens_per_batch, 1
            )
        return 0

    @property
    def scheduler_current(self) -> int:
        units = getattr(getattr(self.cfg, "scheduler", object()), "units", "steps")
        return self.global_train_tokens_seen if units == "tokens" else self.global_step

    @property
    def scheduler_max(self) -> int:
        units = getattr(getattr(self.cfg, "scheduler", object()), "units", "steps")
        return self.max_tokens if units == "tokens" else self.max_steps

    # ----------------------------
    # Checkpointing (simple wrapper around existing Checkpointer)
    # ----------------------------
    def trainer_state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
        }
        # RNG states if torch present
        if torch is not None:
            state["rng"] = {
                "python": random.getstate(),
                "numpy": (np.random.get_state() if np is not None else None),  # type: ignore[union-attr]
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "mps": torch.mps.get_rng_state() if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else None,
            }
        return state

    def load_trainer_state_dict(self, state: Dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.global_train_examples_seen_this_epoch = int(state.get("global_train_examples_seen_this_epoch", 0))
        self.global_train_tokens_seen = int(state.get("global_train_tokens_seen", 0))
        rng = state.get("rng")
        if rng is not None:
            self.restore_rng_state(rng)
        # Reset LR from scheduler (do not trust checkpoint LR)
        try:
            base_lr = getattr(self.cfg.optimizer, "lr", None) or getattr(self.cfg.optimizer, "learning_rate", 0.0)
            new_lr = self.scheduler.get_lr(base_lr, self.scheduler_current, self.scheduler_max)  # type: ignore[attr-defined]
            for group in getattr(self.optimizer, "param_groups", []):
                group["lr"] = new_lr
        except Exception:
            pass

    def restore_rng_state(self, rng: Dict[str, Any]) -> None:
        try:
            random.setstate(rng.get("python"))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            if np is not None and rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])  # type: ignore[assignment]
        except Exception:
            pass
        if torch is not None:  # pragma: no cover - soft dependency
            try:
                torch.set_rng_state(rng.get("torch"))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state(rng["cuda"])  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and rng.get("mps") is not None:
                    torch.mps.set_rng_state(rng["mps"])  # type: ignore[arg-type]
            except Exception:
                pass

    def save_checkpoint(self, name: Optional[str] = None) -> Optional[Path]:  # pragma: no cover - I/O shim
        if self.checkpointer is None:
            return None
        try:
            ckpt_name = name or f"step-{self.global_step}"
            self.checkpointer.save(name=ckpt_name)
            return Path(self.cfg.save_folder) / ckpt_name
        except Exception:
            return None

    # ----------------------------
    # Core train/eval helpers (safe, optional behavior)
    # ----------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None or self.device is None:
            return batch
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                try:
                    out[k] = v.to(self.device)
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out

    def get_labels(self, batch: Dict[str, Any]) -> Any:
        labels = batch["input_ids"].clone()
        label_mask = batch.get("label_mask")
        attention_mask = batch.get("attention_mask")
        instance_mask = batch.get("instance_mask")
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), -100)
        return labels[..., 1:].contiguous()

    def model_forward(self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False):
        if torch is None:
            raise RuntimeError("torch is required to run forward pass")
        m = self.model  # type: ignore[assignment]
        outputs = m(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        )
        logits = getattr(outputs, "logits", outputs)
        logits_for_loss = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = self.get_labels(batch).view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        micro = int(getattr(self.cfg, "device_train_microbatch_size", 0) or 0)
        if micro <= 0:
            return [batch]
        bsz = batch["input_ids"].shape[0]
        if bsz <= micro:
            return [batch]
        micro_batches: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch is not None and hasattr(v, "split") and isinstance(v, torch.Tensor):
                micro_batches[k] = v.split(micro, dim=0)
            elif isinstance(v, list):
                micro_batches[k] = [v[micro * i : micro * (i + 1)] for i in range(math.ceil(bsz / micro))]
            else:
                raise ValueError(f"Unsupported batch field type for key '{k}'")
        return [{k: v[i] for k, v in micro_batches.items()} for i in range(len(micro_batches["input_ids"]))]  # type: ignore[index]

    def train_micro_batch(self, micro_batch: Dict[str, Any], batch_size_in_tokens: int):
        ce_loss, z_loss, logits = self.model_forward(
            micro_batch, compute_z_loss=bool(getattr(self.cfg, "softmax_auxiliary_loss", False)), loss_reduction="sum"
        )
        ce_loss = ce_loss / max(batch_size_in_tokens, 1)
        if getattr(self.cfg, "softmax_auxiliary_loss", False):
            assert z_loss is not None
            z_loss = z_loss / max(batch_size_in_tokens, 1)
            loss = ce_loss + z_loss
        else:
            loss = ce_loss
        # free up
        del logits
        return loss, ce_loss, z_loss

    def train_batch(self, batch: Dict[str, Any]):
        if torch is None:
            raise RuntimeError("torch is required to train")
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = int(batch["input_ids"].numel())
        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not getattr(self.cfg, "softmax_auxiliary_loss", False) else torch.tensor(0.0, device=self.device)
        for idx, micro in enumerate(micro_batches):
            grad_sync_context = nullcontext
            # DDP batch sync placeholder: no-op by default
            with grad_sync_context():
                autocast_device = "mps" if (self.device is not None and self.device.type == "mps") else "cuda"
                try:
                    ctx = torch.autocast(autocast_device, enabled=True, dtype=getattr(self.cfg, "autocast_precision", None))
                except Exception:
                    ctx = nullcontext()
                with ctx:
                    loss, ce_loss, z_loss = self.train_micro_batch(micro, batch_size_in_tokens)
                    ce_batch_loss += ce_loss.detach()
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()
                loss.backward()
        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("torch is required to train")
        metrics: Dict[str, float] = {}
        if hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)
        batch = self._to_device(batch)
        ce_batch_loss, z_batch_loss = self.train_batch(batch)
        # No distributed reduction here; integration point for future dist support
        # Optimizer metrics and gradient clipping if available
        if hasattr(self.optimizer, "clip_grads_and_collect_metrics"):
            try:
                optim_metrics = self.optimizer.clip_grads_and_collect_metrics(self.global_step)
                for k, v in optim_metrics.items():
                    try:
                        metrics[f"optim/{k}"] = float(getattr(v, "item", lambda: v)()) if hasattr(v, "item") else float(v)
                    except Exception:
                        pass
            except Exception:
                pass
        # Adjust LR from scheduler if available
        try:
            base_lr = getattr(self.cfg.optimizer, "lr", None) or getattr(self.cfg.optimizer, "learning_rate", 0.0)
            new_lr = self.scheduler.get_lr(base_lr, self.scheduler_current, self.scheduler_max)  # type: ignore[attr-defined]
            for group in getattr(self.optimizer, "param_groups", []):
                group["lr"] = new_lr
        except Exception:
            try:
                # fallback to simple multiplier step API
                lr_scale = self.scheduler.step()  # type: ignore[attr-defined]
                base_lr = getattr(self.cfg.optimizer, "lr", 0.0) or getattr(self.cfg.optimizer, "learning_rate", 0.0)
                if hasattr(self.optimizer, "set_lr"):
                    self.optimizer.set_lr(base_lr * lr_scale)
                else:
                    for group in getattr(self.optimizer, "param_groups", []):
                        group["lr"] = base_lr * lr_scale
            except Exception:
                pass
        # Step
        if hasattr(self.optimizer, "step"):
            self.optimizer.step()
        # Loss metrics
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        self.cur_train_loss = float(ce_batch_loss.item())
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        try:
            metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        except Exception:
            pass
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = float(z_batch_loss.item())
        return metrics

    # ----------------------------
    # Eval (skeleton)
    # ----------------------------
    def eval_batch(self, batch: Dict[str, Any]):
        if torch is None:
            raise RuntimeError("torch is required for eval")
        try:
            ctx = torch.autocast("cuda", enabled=True, dtype=getattr(self.cfg, "autocast_precision", None))
        except Exception:
            ctx = nullcontext()
        with ctx:
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Any) -> None:
        if torch is None:
            return
        batch = self._to_device(batch)
        with torch.no_grad():
            ce_loss, logits = self.eval_batch(batch)
        if hasattr(evaluator, "update_metrics"):
            evaluator.update_metrics(batch, ce_loss, logits)

    # ----------------------------
    # System metrics & logging
    # ----------------------------
    def system_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if torch is None:
            return out
        try:
            if self.global_step < 3 or self.global_step % 10 == 0:
                peak_gpu_mb = None
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    peak_gpu_mb = float(peak)
                if peak_gpu_mb is not None:
                    out["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        except Exception:
            pass
        return out

    def should_log_this_step(self) -> bool:
        return (self.global_step % int(getattr(self.cfg, "console_log_interval", 50) or 50)) == 0

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        # format and print metrics like OLMo
        log_str = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        log.info(f"{prefix} {log_str}")

    def eval(self) -> Dict[str, float]:
        # implement eval loop over self.evaluators, similar to OLMo
        log.info(f"Starting evaluation")
        total_loss = 0.0
        total_count = 0
        for evaluator in self.evaluators:
            evaluator.reset()
            with torch.no_grad():
                for batch in evaluator:
                    ce_loss, logits = self.eval_batch(batch)
                    total_loss += ce_loss.sum().item()
                    total_count += ce_loss.numel()
                    if hasattr(evaluator, "update_metrics"):
                        evaluator.update_metrics(batch, ce_loss, logits)
        avg_loss = total_loss / total_count if total_count > 0 else 0.0
        log.info(f"Evaluation completed. Avg loss: {avg_loss:.4g}")
        return {"eval/loss": avg_loss}

    # ----------------------------
    # Fit loop (safe, minimal, non-executing by default)
    # ----------------------------
    def check_if_cancelled(self) -> Tuple[bool, int]:
        # implement cancellation logic via time_limit, early_stopping, wandb tag
        if self.cfg.time_limit and (time.monotonic() - self._start_time) > self.cfg.time_limit:
            return True, 0
        if self.cfg.early_stopping and self.global_step >= self.cfg.early_stopping:
            return True, 0
        if wandb.run and wandb.run.resumed:
            return True, 0
        return False, 0

    def fit(self) -> None:  # pragma: no cover - training skeleton
        """Enterprise-style training loop skeleton with safe guards.
        This is designed for learning and code reading, not for execution.
        """
        # Early exit if torch unavailable
        if torch is None:
            return None

        # Optional profiling toggles (no-op by default)
        python_profiler = None
        if bool(getattr(self.cfg, "python_profiling", False)):
            try:
                import cProfile  # type: ignore

                python_profiler = cProfile.Profile()
            except Exception:
                python_profiler = None

        # Set model train mode if possible
        try:
            self.model.train()  # type: ignore[union-attr]
        except Exception:
            pass

        total_steps = int(getattr(self.cfg.scheduler, "total_steps", 0) or self.max_steps or 0)
        base_lr = getattr(self.cfg.optimizer, "lr", None) or getattr(self.cfg.optimizer, "learning_rate", 0.0)

        # Simulated loop (no real data pipeline wired here)
        self._start_time = time.monotonic()
        for step in range(self.global_step, total_steps):
            self.global_step = step + 1
            # Scheduler step (scale LR) fallback
            lr_scale = None
            try:
                lr_scale = self.scheduler.step()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                if lr_scale is not None:
                    if hasattr(self.optimizer, "set_lr"):
                        self.optimizer.set_lr(float(base_lr) * float(lr_scale))
                    else:
                        for g in getattr(self.optimizer, "param_groups", []):
                            g["lr"] = float(base_lr) * float(lr_scale)
                else:
                    new_lr = self.scheduler.get_lr(base_lr, self.scheduler_current, self.scheduler_max)  # type: ignore[attr-defined]
                    for g in getattr(self.optimizer, "param_groups", []):
                        g["lr"] = new_lr
            except Exception:
                pass

            # Placeholder for micro-batch processing, grads, step etc.
            # self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

            # Periodic checkpoint (compatible with current Checkpointer API)
            save_every = int(getattr(self.cfg, "save_interval", 0) or max(1, total_steps // 10))
            if save_every > 0 and self.global_step % save_every == 0:
                self.save_checkpoint(name=f"step-{self.global_step}")

            # Logging hooks
            if self.should_log_this_step():
                metrics = {
                    **self.speed.check(),
                    **self.lrmon.check(),
                    **self.system_metrics(),
                }
                self.log_metrics_to_console("Train", metrics)

            # Optional stop
            stop_at = getattr(self.cfg, "stop_at", None)
            if isinstance(stop_at, int) and self.global_step >= stop_at:
                break

            # Check for cancellation
            cancelled, cancel_code = self.check_if_cancelled()
            if cancelled:
                log.info(f"Training cancelled at step {self.global_step}")
                break

        # Final checkpoint save
        try:
            self.save_checkpoint(name=f"step-{self.global_step}")
        except Exception:
            pass

        # Flush indices_file, reset gc, finish wandb
        self.close()

    def close(self, exit_code: int = 0) -> None:
        # flush indices_file, reset gc, finish wandb
        if self.indices_file:
            try:
                self.indices_file.flush()
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass
        if wandb.run:
            wandb.finish()
        if exit_code != 0:
            os._exit(exit_code)

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - I/O shim
        return None

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(0 if exc_type is None else 1)


__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]
