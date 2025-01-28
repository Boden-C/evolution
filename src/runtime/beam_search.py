from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple


State = Dict[str, "object"]
StepFn = Callable[["object", State], Tuple["object", State]]


@dataclass
class BeamSearch:
    eos_token_id: int
    max_steps: int = 10

    def search(self, start: "object", state: State, step: StepFn) -> Tuple["object", "object"]:
        try:
            import torch  # type: ignore

            seq = start
            scores = torch.zeros((start.size(0),), dtype=torch.float32, device=start.device)  # type: ignore
            for _ in range(self.max_steps):
                next_logits, state = step(seq, state)
                next_id = next_logits.argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, next_id], dim=1)
                if (next_id == self.eos_token_id).all():
                    break
            return seq, scores
        except Exception:
            return start, None  # type: ignore


__all__ = ["BeamSearch"]
