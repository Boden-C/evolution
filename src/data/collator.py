from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataCollator:
    pad_token_id: int = 0

    def __call__(self, batch: List[Dict[str, List[int]]]):
        try:
            import torch  # type: ignore

            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids = []
            attention_mask = []
            for ex in batch:
                ids = ex["input_ids"]
                pad = [self.pad_token_id] * (max_len - len(ids))
                input_ids.append(torch.tensor(ids + pad, dtype=torch.long))
                attention_mask.append(torch.tensor([1] * len(ids) + [0] * len(pad), dtype=torch.float32))
            return {
                "input_ids": torch.stack(input_ids, dim=0),
                "attention_mask": torch.stack(attention_mask, dim=0),
            }
        except Exception:
            return {
                "input_ids": [ex["input_ids"] for ex in batch],
                "attention_mask": [[1] * len(ex["input_ids"]) for ex in batch],
            }


__all__ = ["DataCollator"]
