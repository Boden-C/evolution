from __future__ import annotations

from typing import Dict, List


NAMED_MIXES: Dict[str, List[str]] = {
    "toy": ["tiny_corpus_v1"],
}


def get_named_mix(name: str) -> List[str]:
    try:
        return NAMED_MIXES[name]
    except KeyError:
        raise KeyError(f"Unknown data mix: {name}")


__all__ = ["NAMED_MIXES", "get_named_mix"]
