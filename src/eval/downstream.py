from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    name: str

    def evaluate(self) -> float:  # pragma: no cover - placeholder
        return 0.0


__all__ = ["Task"]
from __future__ import annotations

from typing import Dict


def get_task_registry() -> Dict[str, str]:  # pragma: no cover - placeholder
    return {"mmlu": "Multiple-choice QA (toy)", "lambada": "Completion (toy)"}


__all__ = ["get_task_registry"]
