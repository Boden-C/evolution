from __future__ import annotations

from typing import Iterable, List


def chunk(it: Iterable[int], size: int) -> List[List[int]]:
    buf: List[int] = []
    out: List[List[int]] = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


def get_document_lengths(attention_mask: "object") -> List[int]:  # pragma: no cover - tiny helper
    try:
        import torch  # type: ignore

        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        # treat 0 separators as doc boundaries; sum per doc row if already 2D
        if attention_mask.dim() == 1:
            return [int(attention_mask.sum().item())]
        return [int(row.sum().item()) for row in attention_mask]
    except Exception:
        # Fallback for lists
        if isinstance(attention_mask, list) and attention_mask and isinstance(attention_mask[0], list):
            return [sum(int(x) for x in row) for row in attention_mask]
        return [sum(int(x) for x in attention_mask)]


__all__ = ["chunk", "get_document_lengths"]
