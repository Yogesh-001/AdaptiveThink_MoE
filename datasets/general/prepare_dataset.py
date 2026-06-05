"""Prepare Alpaca-cleaned dataset for general expert training."""

import json
import os
from datasets import load_dataset


def format_sample_to_instruction(sample: dict) -> dict:
    """Convert Alpaca sample to instruction-response format."""
    if sample.get("input") and sample["input"].strip():
        instruction = f"{sample['instruction']}\n\nInput: {sample['input']}"
    else:
        instruction = sample["instruction"]

    return {
        "instruction": instruction,
        "response": sample["output"],
    }


def prepare_dataset(subset_size: int = 500, output_dir: str = None):
    """Download Alpaca-cleaned, filter, format, and save."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading Alpaca-cleaned dataset from HuggingFace...")
    ds = load_dataset("yahma/alpaca-cleaned")
    print(f"Dataset: {ds}")

    # Filter out very short and very long responses
    all_samples = []
    skipped_short = 0
    skipped_long = 0

    for sample in ds["train"]:
        if len(sample["output"]) < 20:
            skipped_short += 1
            continue
        if len(sample["output"]) > 2000:
            skipped_long += 1
            continue
        all_samples.append(format_sample_to_instruction(sample))

    print(f"\nFiltering results:")
    print(f"  Skipped (too short <20 chars): {skipped_short}")
    print(f"  Skipped (too long >2000 chars): {skipped_long}")
    print(f"  Kept: {len(all_samples)}")

    # Sample evenly for diversity
    step = max(1, len(all_samples) // subset_size)
    subset = [all_samples[i] for i in range(0, len(all_samples), step)][:subset_size]

    # Save full dataset
    full_path = os.path.join(output_dir, "alpaca_formatted_full.json")
    with open(full_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nSaved full dataset: {full_path} ({len(all_samples)} samples)")

    # Save subset
    subset_path = os.path.join(output_dir, "alpaca_formatted_subset.json")
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
