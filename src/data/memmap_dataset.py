from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Union
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type check only
    import numpy as np


class MemmapDataset:
    """A minimal numpy.memmap-backed dataset.

    Supports two layouts:
      1) Fixed-size records stored contiguously in a single binary file, with a
         given dtype and shape per record.
      2) Variable-length byte records with a companion JSONL index file containing
         {"offset": int, "length": int} per line.

    This is intentionally simple but practical. It is not a drop-in OLMo port.
    """

    def __init__(
        self,
        data_path: Union[str, os.PathLike[str]],
        *,
        dtype: Optional["np.dtype"] = None,
        shape: Optional[Tuple[int, ...]] = None,
        index_path: Optional[Union[str, os.PathLike[str]]] = None,
        mode: str = "r",
    ) -> None:
        self.data_path = str(data_path)
        self.index_path = str(index_path) if index_path is not None else None
        self.mode = mode
        self._fixed = dtype is not None and shape is not None and index_path is None
        # Import numpy lazily to keep optional dependency.
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "NumPy is required for MemmapDataset. Please install numpy."
            ) from e
        self._np = np
        self._dtype = self._np.dtype(dtype) if dtype is not None else None
        self._shape = tuple(shape) if shape is not None else None

        if self._fixed:
            # Fixed-size records. Compute number of records from file size.
            itemsize = int(self._dtype.itemsize)  # type: ignore[union-attr]
            n_elem = int(self._np.prod(self._shape))  # type: ignore[arg-type]
            record_bytes = itemsize * n_elem
            file_size = os.path.getsize(self.data_path)
            if file_size % record_bytes != 0:
                raise ValueError("File size is not a multiple of record size")
            num_records = file_size // record_bytes
            self._memmap = self._np.memmap(self.data_path, dtype=self._dtype, mode=mode)
            self._num_records = int(num_records)
        else:
            # Variable-length records via JSONL index.
            if self.index_path is None:
                raise ValueError("index_path is required for variable-length records")
            self._memmap = self._np.memmap(self.data_path, dtype=self._np.uint8, mode=mode)
            self._offsets: List[int] = []
            self._lengths: List[int] = []
            with open(self.index_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._offsets.append(int(rec["offset"]))
                    self._lengths.append(int(rec["length"]))
            self._num_records = len(self._offsets)

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, idx: int):  # type: ignore[override]
        if idx < 0 or idx >= self._num_records:
            raise IndexError(idx)
        if self._fixed:
            # Using view to avoid copies, then reshape to per-record shape.
            n_elem = int(self._np.prod(self._shape))  # type: ignore[arg-type]
            start = idx * n_elem
            end = start + n_elem
            arr = self._memmap[start:end].reshape(self._shape)  # type: ignore[index]
            return self._np.array(arr)  # materialize small slice
        else:
            off = self._offsets[idx]
            ln = self._lengths[idx]
            view = self._memmap[off : off + ln]
            return bytes(view)

    def __iter__(self) -> Iterator[Union["np.ndarray", bytes]]:  # pragma: no cover - simple
        for i in range(len(self)):
            yield self[i]


__all__ = ["MemmapDataset"]
