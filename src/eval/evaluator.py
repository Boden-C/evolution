from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Evaluator:
    tasks: List[str]

    def evaluate(self) -> Dict[str, float]:  # pragma: no cover - placeholder
        return {t: 0.0 for t in self.tasks}


__all__ = ["Evaluator"]
