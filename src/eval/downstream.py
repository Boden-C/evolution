from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Task:
    name: str
    run: Callable[[], float]

    def evaluate(self) -> float:  # pragma: no cover - placeholder
        return float(self.run())


def get_task_registry() -> Dict[str, Task]:  # pragma: no cover - placeholder
    return {
        "mmlu": Task(name="mmlu", run=lambda: 0.0),
        "lambada": Task(name="lambada", run=lambda: 0.0),
    }


__all__ = ["Task", "get_task_registry"]
