from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple


State = Dict[str, "object"]
StepFn = Callable[["object", State], Tuple["object", State]]


class Constraint:
    def apply(self, seq: "object", logits: "object") -> "object":  # pragma: no cover - torch optional
        return logits


class RepeatedNGramBlockingConstraint(Constraint):
    def __init__(self, n: int = 3):
        self.n = max(1, int(n))

    def apply(self, seq: "object", logits: "object") -> "object":  # pragma: no cover - torch optional
        try:
            import torch  # type: ignore

            B = seq.size(0)
            if seq.size(1) < self.n:
                return logits
            for b in range(B):
                seq_b = [int(x) for x in seq[b].tolist()]
                if len(seq_b) < self.n:
                    continue
                prefix = seq_b[-(self.n - 1) :] if self.n > 1 else []
                banned: List[int] = []
                for i in range(len(seq_b) - self.n + 1):
                    if self.n == 1 or seq_b[i : i + self.n - 1] == prefix:
                        banned.append(int(seq_b[i + self.n - 1]))
                if banned:
                    logits[b, banned] = float("-inf")
            return logits
        except Exception:
            return logits


class Sampler:
    def sample(self, logits: "object") -> Tuple["object", "object"]:  # ids, logprobs
        return logits, logits


class GreedySampler(Sampler):
    def sample(self, logits: "object") -> Tuple["object", "object"]:
        try:
            import torch  # type: ignore

            ids = logits.argmax(dim=-1)
            logp = logits.log_softmax(dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            return ids, logp
        except Exception:  # pragma: no cover - fallback
            return logits, logits


class TopKSampler(Sampler):
    def __init__(self, k: int = 50):
        self.k = max(1, int(k))

    def sample(self, logits: "object") -> Tuple["object", "object"]:
        try:
            import torch  # type: ignore

            k = min(self.k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            probs = vals.softmax(dim=-1)
            choice = torch.multinomial(probs, 1).squeeze(-1)
            ids = idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            logp = (probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1) + 1e-40).log()
            return ids, logp
        except Exception:  # pragma: no cover
            return logits, logits


class TopPSampler(Sampler):
    def __init__(self, p: float = 0.9):
        self.p = float(p)

    def sample(self, logits: "object") -> Tuple["object", "object"]:
        try:
            import torch  # type: ignore

            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = sorted_logits.softmax(dim=-1)
            cumprobs = probs.cumsum(dim=-1)
            mask = cumprobs <= self.p
            # Ensure at least one token
            mask[..., 0] = True
            # Mask logits outside nucleus
            filtered = torch.where(mask, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
            nucleus_probs = filtered.softmax(dim=-1)
            choice = torch.multinomial(nucleus_probs, 1).squeeze(-1)
            ids = sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            logp = (nucleus_probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1) + 1e-40).log()
            return ids, logp
        except Exception:  # pragma: no cover
            return logits, logits


class FinalScorer:
    def score(self, token_logprobs: "object", lengths: "object") -> "object":
        return token_logprobs.sum(dim=-1)


class LengthNormalizedFinalScorer(FinalScorer):
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def score(self, token_logprobs: "object", lengths: "object") -> "object":
        try:
            import torch  # type: ignore

            denom = torch.clamp(lengths.to(token_logprobs.dtype) ** self.alpha, min=1.0)
            return token_logprobs.sum(dim=-1) / denom
        except Exception:  # pragma: no cover
            return token_logprobs


@dataclass
class BeamSearch:
    eos_token_id: int
    max_steps: int = 10
    beam_size: int = 1
    min_steps: int = 0
    sampler: Optional[Sampler] = None
    scorer: Optional[FinalScorer] = None
    constraints: Optional[List[Constraint]] = None

    def search(self, start: "object", state: State, step: StepFn) -> Tuple["object", "object"]:
        try:
            import torch  # type: ignore

            sampler = self.sampler or GreedySampler()
            scorer = self.scorer or FinalScorer()
            constraints = self.constraints or []

            B, L0 = int(start.size(0)), int(start.size(1))  # type: ignore
            device = start.device  # type: ignore
            beam = self.beam_size

            seqs = start.unsqueeze(1).expand(B, beam, L0).contiguous()  # [B, beam, L]
            token_logps = torch.zeros((B, beam, 0), dtype=torch.float32, device=device)
            alive = torch.ones((B, beam), dtype=torch.bool, device=device)
            finished_seq: Optional[torch.Tensor] = None
            finished_scores: Optional[torch.Tensor] = None

            cur_inputs = seqs.view(B * beam, L0)
            cur_state = state

            for t in range(self.max_steps):
                logits, cur_state = step(cur_inputs, cur_state)
                logits = logits.view(B, beam, -1)
                for cons in constraints:
                    logits = cons.apply(seqs.view(B * beam, -1), logits.view(B * beam, -1)).view(B, beam, -1)
                if t < self.min_steps:
                    logits[..., self.eos_token_id] = float("-inf")

                ids, logp = sampler.sample(logits.view(B * beam, -1))
                ids = ids.view(B, beam)
                logp = logp.view(B, beam)

                seqs = torch.cat([seqs, ids.unsqueeze(-1)], dim=-1)
                token_logps = torch.cat([token_logps, logp.unsqueeze(-1)], dim=-1)

                just_finished = ids.eq(self.eos_token_id)
                alive = alive & (~just_finished)

                # Gather candidates and pick top beams per batch
                lengths = (seqs != self.eos_token_id).sum(dim=-1)
                scores = scorer.score(token_logps, lengths)

                # If a hypothesis finished this step, keep it aside
                if finished_seq is None:
                    finished_seq = seqs.clone()
                    finished_scores = scores.clone()
                else:
                    keep = scores > finished_scores
                    finished_seq = torch.where(keep.unsqueeze(-1), seqs, finished_seq)
                    finished_scores = torch.where(keep, scores, finished_scores)

                if not alive.any():
                    break

                # Prepare next inputs
                top_scores, top_idx = torch.topk(scores, k=beam, dim=1)
                b_idx = torch.arange(B, device=device).unsqueeze(-1)
                seqs = seqs[b_idx, top_idx]
                token_logps = token_logps[b_idx, top_idx]
                alive = alive[b_idx, top_idx]
                cur_inputs = seqs.view(B * beam, -1)

            # Finalize
            if finished_seq is None or finished_scores is None:
                return start, torch.zeros((B,), dtype=torch.float32, device=device)
            # Take best beam per batch
            best_scores, best_idx = torch.max(finished_scores, dim=1)
            b_idx = torch.arange(B, device=device)
            best_seq = finished_seq[b_idx, best_idx]
            return best_seq, best_scores
        except Exception:
            return start, None  # type: ignore


__all__ = [
    "Constraint",
    "RepeatedNGramBlockingConstraint",
    "Sampler",
    "GreedySampler",
    "TopKSampler",
    "TopPSampler",
    "FinalScorer",
    "LengthNormalizedFinalScorer",
    "BeamSearch",
]
