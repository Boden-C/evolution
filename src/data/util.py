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


__all__ = ["chunk"]
