from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..aliases import PathOrStr
from ..exceptions import TokenizerError


@dataclass
class TokenizerInfo:
    name_or_path: Optional[PathOrStr]
    vocab_size: int


class Tokenizer:
    """Minimal whitespace tokenizer with EOS/pad handling and truncation.

    This intentionally avoids heavy deps while providing a stable surface:
    - encode/decode with optional EOS
    - batch encode
    - truncation with left/right direction
    - simple save/load shims for drop-in replacement later
    """

    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __init__(
        self,
        name_or_path: Optional[PathOrStr] = None,
        vocab_size: int = 50280,
        truncate_to: Optional[int] = None,
        truncate_direction: str = "right",
    ):
        self._info = TokenizerInfo(name_or_path=name_or_path, vocab_size=vocab_size)
        self._vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        self._inv_vocab = {self.pad_token_id: self.pad_token, self.eos_token_id: self.eos_token}
        self._next_id = max(self._inv_vocab) + 1
        self._truncate_to = truncate_to
        self._truncate_direction = truncate_direction

    @property
    def vocab_size(self) -> int:
        return max(self._info.vocab_size, len(self._vocab))

    def _maybe_truncate(self, ids: List[int], max_length: Optional[int], direction: Optional[str]) -> List[int]:
        if max_length is None:
            max_length = self._truncate_to
        if direction is None:
            direction = self._truncate_direction
        if max_length is None or max_length <= 0:
            return ids
        if len(ids) <= max_length:
            return ids
        if direction not in {"left", "right"}:
            raise TokenizerError("truncate_direction must be 'left' or 'right'")
        return ids[-max_length:] if direction == "left" else ids[:max_length]

    def add_special_tokens(self, ids: Sequence[int], add_eos_if_not_present: bool = True) -> List[int]:
        out = list(ids)
        if add_eos_if_not_present and (len(out) == 0 or out[-1] != self.eos_token_id):
            out.append(self.eos_token_id)
        return out

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:  # parity with common APIs
        return 1 if not is_pair else 2  # simple scheme: EOS per sequence

    def encode(
        self,
        text: str,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        truncation_direction: Optional[str] = None,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TokenizerError("text must be a string")
        ids: List[int] = []
        for tok in text.strip().split():
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._inv_vocab[self._next_id] = tok
                self._next_id += 1
            ids.append(self._vocab[tok])
        ids = self._maybe_truncate(ids, max_length, truncation_direction)
        if add_eos:
            ids = self.add_special_tokens(ids, add_eos_if_not_present=True)
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        add_eos: bool = False,
        max_length: Optional[int] = None,
        truncation_direction: Optional[str] = None,
    ) -> List[List[int]]:
        return [
            self.encode(t, add_eos=add_eos, max_length=max_length, truncation_direction=truncation_direction)
            for t in texts
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            if skip_special_tokens and int(i) in {self.pad_token_id, self.eos_token_id}:
                continue
            toks.append(self._inv_vocab.get(int(i), "<unk>"))
        return " ".join(toks).strip()

    def save_pretrained(self, path: PathOrStr) -> None:  # pragma: no cover - I/O shim
        # Minimal metadata-only save to ease future migration to HF/tokenizers.
        import json, os

        os.makedirs(str(path), exist_ok=True)
        meta = {
            "name_or_path": str(self._info.name_or_path) if self._info.name_or_path is not None else None,
            "vocab": self._vocab,
            "truncate_to": self._truncate_to,
            "truncate_direction": self._truncate_direction,
        }
        with open(os.path.join(str(path), "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def from_pretrained(cls, path: PathOrStr) -> "Tokenizer":  # pragma: no cover - I/O shim
        return cls.from_file(path)

    @classmethod
    def from_file(cls, path: PathOrStr) -> "Tokenizer":  # pragma: no cover - I/O shim
        import json, os
        meta_path = os.path.join(str(path), "tokenizer.json")
        if not os.path.exists(meta_path):
            # Fallback to basic instance with name recorded
            return cls(name_or_path=path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tok = cls(
            name_or_path=meta.get("name_or_path"),
            vocab_size=max(50280, len(meta.get("vocab", {}))),
            truncate_to=meta.get("truncate_to"),
            truncate_direction=meta.get("truncate_direction", "right"),
        )
        # Restore vocab
        vocab = meta.get("vocab", {})
        tok._vocab = {k: int(v) for k, v in vocab.items()}
        tok._inv_vocab = {int(v): k for k, v in vocab.items()}
        tok._next_id = (max(tok._inv_vocab) + 1) if tok._inv_vocab else 0
        return tok

    @classmethod
    def from_checkpoint(cls, folder: PathOrStr) -> "Tokenizer":  # pragma: no cover - I/O shim
        return cls.from_file(folder)

    @classmethod
    def from_train_config(cls, name_or_path: Optional[PathOrStr], truncation_direction: str = "right") -> "Tokenizer":
        return cls(name_or_path=name_or_path, truncate_direction=truncation_direction)

    def ensure_vocab_size(self, expected: int) -> None:
        if expected > self.vocab_size:
            self._info.vocab_size = expected


__all__ = ["Tokenizer", "TokenizerInfo"]
