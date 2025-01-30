from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..config import PaddingDirection, TrainConfig


@dataclass
class DataCollator:
    pad_direction: "PaddingDirection"
    pad_token_id: int

    @classmethod
    def from_train_config(cls, config: "TrainConfig") -> "DataCollator":
        return cls(pad_direction=config.data.pad_direction, pad_token_id=config.model.pad_token_id)

    def __call__(self, items: Union[List[Dict[str, Any]], List[Any]]) -> Dict[str, Any]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))

        all_input_ids: List[torch.Tensor] = []
        all_attention_mask: List[torch.Tensor] = []
        all_attention_bias: List[torch.Tensor] = []
        all_label_mask: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []
        all_metadata: List[Any] = []
        all_instance_mask: List[torch.Tensor] = []
        all_doc_lens: List[torch.Tensor] = []
        all_max_doc_lens: List[int] = []

        max_docs = max((len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items))

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # input_ids
            all_input_ids.append(
                F.pad(input_ids.to(dtype=torch.long), pad_shape, value=self.pad_token_id)
            )

            # attention_mask
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(attention_mask.to(dtype=torch.float), pad_shape, value=0.0)
                )

            # attention_bias (shape to 3D and pad both dims)
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(attention_bias, pad_shape + pad_shape, value=pad_value)
                )

            # label_mask
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(label_mask.to(dtype=torch.bool), pad_shape, value=False)
                )

            # index
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # instance_mask
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # doc_lens
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                import torch as _torch

                if not isinstance(doc_lens, _torch.Tensor):
                    doc_lens = _torch.tensor(doc_lens, dtype=_torch.long)
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

            # metadata
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_instance_mask:
            out["instance_mask"] = torch.stack(all_instance_mask)
        if all_doc_lens:
            out["doc_lens"] = torch.stack(all_doc_lens)
        if all_max_doc_lens:
            out["max_doc_lens"] = all_max_doc_lens
        if all_metadata:
            out["metadata"] = all_metadata
        return out


@dataclass
class CustomDatasetDataCollator(DataCollator):
    input_id_field: str = "input_ids"
    attention_mask_field: Optional[str] = None
    attention_bias_field: Optional[str] = None
    label_mask_field: Optional[str] = None
    index_field: Optional[str] = None
    instance_mask_field: Optional[str] = None
    doc_lens_field: Optional[str] = None
    metadata_field: Optional[str] = None

    def _relabel_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "input_ids": item[self.input_id_field],
            "attention_mask": item[self.attention_mask_field] if self.attention_mask_field else None,
            "attention_bias": item[self.attention_bias_field] if self.attention_bias_field else None,
            "label_mask": item[self.label_mask_field] if self.label_mask_field else None,
            "index": item[self.index_field] if self.index_field else None,
            "instance_mask": item[self.instance_mask_field] if self.instance_mask_field else None,
            "metadata": item[self.metadata_field] if self.metadata_field else None,
        }
        if self.doc_lens_field:
            out["doc_lens"] = item.__getitem__(self.doc_lens_field)
        return out

    def _relabel_fields(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._relabel_item(x) for x in items]

    def __call__(self, items: Union[List[Dict[str, Any]], List[Any]]) -> Dict[str, Any]:
        import torch  # type: ignore

        if not isinstance(items[0], torch.Tensor):
            items = self._relabel_fields(items)  # type: ignore
        return super().__call__(items)  # type: ignore


__all__ = ["DataCollator", "CustomDatasetDataCollator"]
