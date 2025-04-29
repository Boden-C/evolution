"""
Script for preparing conversational fine-tuning data for Evolution series models.
Handles tokenization, filtering, and serialization for efficient training.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track


# Import Evolution model tokenizer and CLI utilities
from src.tokenizer import Tokenizer
from src.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(opts) -> None:
    """
    Main entry point for preparing conversational fine-tuning data.
    Handles tokenizer loading, dataset processing, filtering, and serialization.
    """
    # Load tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    # Load dataset (replace with your dataset path or identifier)
    dataset = ds.load_dataset(
        opts.dataset,
        split=opts.split,
    )

    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=opts.num_proc,
    )

    log.info("Filtering dataset...")
    n = len(dataset)
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset, description="Counting tokens"):
        if len(ex["input_ids"]) != opts.seq_len:
            log.warning(f"Example with unexpected sequence length: {len(ex['input_ids'])}")
        total_tokens += len(ex["input_ids"])
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(dataset, description="Writing to memmap"):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    """
    Filter out examples with no labeled tokens.
    """
    return example["n_labels"] > 0


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    """
    Tokenize and mask conversational example for supervised fine-tuning.
    Args:
        example: dict with 'messages' (list of dicts with 'role' and 'content')
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
    Returns:
        dict with 'input_ids', 'label_mask', and 'n_labels'
    """
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content = msg["content"].strip() + tokenizer.eos_token + "\n"
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            label_mask += [True] * len(content_tokens)
            # Mask out the last '\n' (not part of label)
            if len(content_tokens) > 1 and content_tokens[-2] == tokenizer.eos_token_id:
                label_mask[-1] = False
        else:
            content = msg["content"].strip() + "\n"
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    # Truncate and pad to max_seq_len
    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    """
    Argument parser for data preparation script.
    """
    parser = ArgumentParser(description="Prepare conversational dataset for Evolution model series.")
    parser.add_argument("output_dir", type=str, help="Directory to save the results to.")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset path or identifier.", default="allenai/tulu-v2-sft-mixture")
    parser.add_argument("--split", type=str, help="Dataset split to use.", default="train")
    parser.add_argument("-t", "--tokenizer", type=str, help="Tokenizer path or identifier.", default=Path(__file__).parent / "tokenizers" / "default_tokenizer.json")
    parser.add_argument("-s", "--seq-len", type=int, help="Max sequence length.", default=2048)
    parser.add_argument("--eos", type=int, help="EOS token ID.", default=50279)
    parser.add_argument("--pad", type=int, help="PAD token ID.", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="Number of workers.", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    try:
        main(opts)
    except Exception as e:
        log.error(f"Data preparation failed: {e}", exc_info=True)
