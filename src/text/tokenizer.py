from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..aliases import PathOrStr
from ..exceptions import TokenizerError


@dataclass
class TokenizerInfo:
    name_or_path: Optional[PathOrStr]
    vocab_size: int


class Tokenizer:
    def __init__(self, name_or_path: Optional[PathOrStr] = None, vocab_size: int = 50280):
        self._info = TokenizerInfo(name_or_path=name_or_path, vocab_size=vocab_size)
        self._vocab = {"<pad>": 0, "<eos>": 2}
        self._inv_vocab = {0: "<pad>", 2: "<eos>"}
        self._next_id = 3

    @property
    def vocab_size(self) -> int:
        return max(self._info.vocab_size, len(self._vocab))

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TokenizerError("text must be a string")
        ids: List[int] = []
        for tok in text.strip().split():
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._inv_vocab[self._next_id] = tok
                self._next_id += 1
            ids.append(self._vocab[tok])
        if add_eos:
            ids.append(self._vocab["<eos>"])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self._inv_vocab.get(int(i), "<unk>") for i in ids]
        return " ".join(toks).strip()

    def save_pretrained(self, path: PathOrStr) -> None:  # pragma: no cover - I/O shim
        return None

    @classmethod
    def from_pretrained(cls, path: PathOrStr) -> "Tokenizer":  # pragma: no cover - I/O shim
        return cls(name_or_path=path)


__all__ = ["Tokenizer", "TokenizerInfo"]
