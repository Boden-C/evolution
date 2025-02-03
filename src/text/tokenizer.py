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
    """Whitespace tokenizer with EOS/pad handling, truncation, and lightweight I/O.

    Goals:
    - Stable surface: encode/decode, batch encode, truncation, special tokens.
    - No heavy dependencies; simple save/load for future migration.
    - Enterprise baseline: explicit validation, predictable behavior, concise API.
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
        pad_token_id: Optional[int] = None,
    ):
        self._info = TokenizerInfo(name_or_path=name_or_path, vocab_size=vocab_size)
        self._vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        self._inv_vocab = {self.pad_token_id: self.pad_token, self.eos_token_id: self.eos_token}
        self._next_id = max(self._inv_vocab) + 1
        self._truncate_to = truncate_to
        # Normalize and validate truncation direction early.
        if truncate_direction not in {"left", "right"}:
            raise TokenizerError("truncate_direction must be 'left' or 'right'")
        self._truncate_direction = truncate_direction
        # Allow instance-level override of pad token id; default to eos if None per common practice.
        if pad_token_id is not None:
            # Remap pad token id if custom provided.
            self.pad_token_id = int(pad_token_id)
            self._vocab[self.pad_token] = self.pad_token_id
            self._inv_vocab[self.pad_token_id] = self.pad_token

    @property
    def vocab_size(self) -> int:
        return max(self._info.vocab_size, len(self._vocab))

    @property
    def eos_token_str(self) -> str:
        """Return EOS token string."""
        return self.eos_token

    @property
    def pad_token_str(self) -> str:
        """Return pad token string."""
        return self.pad_token

    # Internal helpers
    def _truncate_ids(self, ids: List[int], max_length: Optional[int], direction: Optional[str]) -> List[int]:
        """Truncate token ids according to settings. No-op if length within limit."""
        if max_length is None:
            max_length = self._truncate_to
        if direction is None:
            direction = self._truncate_direction
        if max_length is None or max_length <= 0 or len(ids) <= max_length:
            return ids
        if direction == "left":
            return ids[-max_length:]
        if direction == "right":
            return ids[:max_length]
        raise TokenizerError("truncate_direction must be 'left' or 'right'")

    # Backward-compat shim
    def _maybe_truncate(self, ids: List[int], max_length: Optional[int], direction: Optional[str]) -> List[int]:
        return self._truncate_ids(ids, max_length, direction)

    def add_eos_if_missing(self, ids: Sequence[int]) -> List[int]:
        """Append EOS if not present."""
        out = list(ids)
        if not out or out[-1] != self.eos_token_id:
            out.append(self.eos_token_id)
        return out

    def add_special_tokens(self, input_ids: Sequence[int]) -> List[int]:  # parity with common APIs
        """Append sequence-level special tokens (EOS). Idempotent."""
        return self.add_eos_if_missing(input_ids)

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:  # parity with common APIs
        return 1 if not is_pair else 2  # EOS per sequence

    def encode(
        self,
        text: str,
        add_eos: bool = False,
        add_special_tokens: Optional[bool] = None,
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
        # Truncation before specials; allow override per call.
        ids = self._truncate_ids(ids, max_length, truncation_direction)
        # Determine whether to add specials (compat with add_eos).
        add_specials = add_eos if add_special_tokens is None else bool(add_special_tokens)
        if add_specials:
            ids = self.add_special_tokens(ids)
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        add_eos: bool = False,
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        truncation_direction: Optional[str] = None,
    ) -> List[List[int]]:
        # Compute effective truncation taking special tokens into account.
        effective_max = max_length if max_length is not None else self._truncate_to
        add_specials = add_eos if add_special_tokens is None else bool(add_special_tokens)
        if effective_max is not None and add_specials:
            effective_max = max(0, effective_max - self.num_special_tokens_to_add(False))
        out: List[List[int]] = []
        for t in texts:
            ids = self.encode(
                t,
                add_eos=False,  # handled here
                add_special_tokens=False,
                max_length=effective_max,
                truncation_direction=truncation_direction,
            )
            if add_specials:
                ids = self.add_special_tokens(ids)
            out.append(ids)
        return out

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            ii = int(i)
            if skip_special_tokens and ii in {self.pad_token_id, self.eos_token_id}:
                continue
            toks.append(self._inv_vocab.get(ii, "<unk>"))
        return " ".join(toks).strip()

    def save_pretrained(self, path: PathOrStr) -> None:  # pragma: no cover - I/O shim
        # Minimal metadata-only save to ease future migration to full tokenizers.
        import json, os

        os.makedirs(str(path), exist_ok=True)
        meta = {
            "name_or_path": str(self._info.name_or_path) if self._info.name_or_path is not None else None,
            "vocab": self._vocab,
            "truncate_to": self._truncate_to,
            "truncate_direction": self._truncate_direction,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
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
            pad_token_id=meta.get("pad_token_id"),
        )
        # Restore vocab and special ids.
        vocab = meta.get("vocab", {})
        tok._vocab = {k: int(v) for k, v in vocab.items()}
        tok._inv_vocab = {int(v): k for k, v in vocab.items()}
        # Respect saved special ids if present.
        if "eos_token_id" in meta:
            tok.eos_token_id = int(meta["eos_token_id"])  # instance override
        if "pad_token_id" in meta:
            tok.pad_token_id = int(meta["pad_token_id"])  # instance override
        tok._next_id = (max(tok._inv_vocab) + 1) if tok._inv_vocab else 0
        return tok

    @classmethod
    def from_checkpoint(cls, folder: PathOrStr) -> "Tokenizer":  # pragma: no cover - I/O shim
        return cls.from_file(folder)

    @classmethod
    def from_train_config(cls, name_or_path: Optional[PathOrStr], truncation_direction: str = "right") -> "Tokenizer":
        # Lightweight factory aligned with internal config usage.
        return cls(name_or_path=name_or_path, truncate_direction=truncation_direction)

    def ensure_vocab_size(self, expected: int) -> None:
        if expected > self.vocab_size:
            self._info.vocab_size = expected

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Tokenizer(name_or_path={self._info.name_or_path!r}, vocab_size={self.vocab_size}, "
            f"truncate_to={self._truncate_to}, truncate_direction={self._truncate_direction!r})"
        )


__all__ = ["Tokenizer", "TokenizerInfo"]
