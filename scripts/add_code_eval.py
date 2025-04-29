"""
Generates code evaluation datasets for perplexity analysis.
Supports HumanEval and MBPP datasets.
"""
import os
import sys

import pandas as pd
from datasets import Dataset, load_dataset


def create_raw_dataset(
    dataset: Dataset,
    id_key: str,
    prompt_key: str,
    answer_key: str,
    save_to: str
) -> None:
    instances = []
    for instance in dataset:
        updated_instance = {
            "id": instance.pop(id_key),
            "text": instance.pop(prompt_key) + instance.pop(answer_key),
            "metadata": instance
        }
        instances.append(updated_instance)

    df = pd.DataFrame.from_records(instances)
    df.to_json(save_to, lines=True, compression="gzip", orient="records")


def run(outdir: str) -> None:
    # Load evaluation datasets for code generation benchmarks
    humaneval = load_dataset("openai_humaneval")
    create_raw_dataset(
        humaneval["test"],
        id_key="task_id",
        prompt_key="prompt",
        answer_key="canonical_solution",
        save_to=os.path.join(outdir, "humaneval_test.jsonl.gz")
    )

    mbpp = load_dataset("mbpp")
    create_raw_dataset(
        mbpp["validation"],
        id_key="task_id",
        prompt_key="text",
        answer_key="code",
        save_to=os.path.join(outdir, "mbpp_validation.jsonl.gz")
    )
    create_raw_dataset(
        mbpp["test"],
        id_key="task_id",
        prompt_key="text",
        answer_key="code",
        save_to=os.path.join(outdir, "mbpp_test.jsonl.gz")
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Output directory argument required.")
    run(sys.argv[1])
