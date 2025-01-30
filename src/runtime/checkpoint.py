from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..aliases import PathOrStr


def _atomic_write_dir(tmp_dir: str, final_dir: str) -> None:
    # On Windows, shutil.rmtree + rename handling needs care; fallback to replace.
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(tmp_dir, final_dir)


@dataclass
class Checkpointer:
    save_folder: PathOrStr

    def _checkpoint_dir(self, name: str) -> str:
        return os.path.join(str(self.save_folder), name)

    def save(self, name: str = "model", state_dict: dict[str, Any] | None = None) -> str:
        os.makedirs(str(self.save_folder), exist_ok=True)
        final_dir = self._checkpoint_dir(name)
        tmp_dir = tempfile.mkdtemp(prefix=f"{name}.tmp.", dir=str(self.save_folder))
        meta = {"name": name}
        # Write metadata first
        with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        # Optional model state (store as JSON for this minimal baseline)
        if state_dict is not None:
            with open(os.path.join(tmp_dir, "state.json"), "w", encoding="utf-8") as f:
                json.dump(state_dict, f)
        # Mark complete via atomic dir replace
        _atomic_write_dir(tmp_dir, final_dir)
        # Update latest symlink/marker
        try:
            latest = os.path.join(str(self.save_folder), "latest.txt")
            with open(latest, "w", encoding="utf-8") as f:
                f.write(name)
        except Exception:
            pass
        return final_dir

    def load(self, name: str = "model") -> dict[str, Any] | None:
        folder = self._checkpoint_dir(name)
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            return None
        result: dict[str, Any] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            result["meta"] = json.load(f)
        state_path = os.path.join(folder, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                result["state"] = json.load(f)
        return result

    def find_latest(self) -> Optional[str]:
        latest_marker = os.path.join(str(self.save_folder), "latest.txt")
        if os.path.exists(latest_marker):
            try:
                with open(latest_marker, "r", encoding="utf-8") as f:
                    name = f.read().strip()
                if name:
                    return name
            except Exception:
                pass
        # Fallback: scan directories ordered by mtime
        try:
            entries = [
                (e, os.path.getmtime(os.path.join(str(self.save_folder), e)))
                for e in os.listdir(str(self.save_folder))
                if os.path.isdir(os.path.join(str(self.save_folder), e))
            ]
            if not entries:
                return None
            entries.sort(key=lambda x: x[1], reverse=True)
            return entries[0][0]
        except Exception:
            return None

    def prune(self, keep: int, predicate: Optional[Callable[[str], bool]] = None) -> None:
        try:
            entries = [
                (e, os.path.getmtime(os.path.join(str(self.save_folder), e)))
                for e in os.listdir(str(self.save_folder))
                if os.path.isdir(os.path.join(str(self.save_folder), e)) and (predicate(e) if predicate else True)
            ]
            entries.sort(key=lambda x: x[1], reverse=True)
            for name, _ in entries[keep:]:
                shutil.rmtree(os.path.join(str(self.save_folder), name), ignore_errors=True)
        except Exception:
            return


__all__ = ["Checkpointer"]
