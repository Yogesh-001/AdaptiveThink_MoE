"""Prepare MBPP dataset for code expert training."""

import json
import os
from datasets import load_dataset


def format_sample_to_instruction(sample: dict) -> dict:
    """Convert MBPP sample to instruction-response format."""
    test_cases = "\n".join(sample["test_list"])
    instruction = (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {sample['text']}\n\n"
        f"Test cases:\n{test_cases}"
    )
    return {
        "instruction": instruction,
        "response": sample["code"],
        "task_id": sample["task_id"],
    }


def prepare_dataset(subset_size: int = 500, output_dir: str = None):
    """Download MBPP, format to instruction-response pairs, and save."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading MBPP dataset from HuggingFace...")
    ds = load_dataset("google-research-datasets/mbpp")
    print(f"Dataset splits: {ds}")

    # Combine all splits for maximum training data
    all_samples = []
    for split in ["train", "test", "validation"]:
        for sample in ds[split]:
            all_samples.append(format_sample_to_instruction(sample))

    print(f"Total formatted samples: {len(all_samples)}")
    subset = all_samples[:subset_size]

    # Save full dataset
    full_path = os.path.join(output_dir, "mbpp_formatted_full.json")
    with open(full_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Saved full dataset: {full_path} ({len(all_samples)} samples)")

    # Save subset
    subset_path = os.path.join(output_dir, "mbpp_formatted_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved subset: {subset_path} ({len(subset)} samples)")

    print(f"\n{'=' * 60}")
    print("FORMATTED EXAMPLE:")
    print(f"{'=' * 60}")
    print(f"\n[INSTRUCTION]:\n{subset[0]['instruction']}")
    print(f"\n[RESPONSE]:\n{subset[0]['response']}")

    return all_samples, subset


if __name__ == "__main__":
    prepare_dataset()
