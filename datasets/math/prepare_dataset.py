"""Prepare GSM8K dataset for math expert training."""

import json
import os
from datasets import load_dataset


def format_sample_to_instruction(sample: dict) -> dict:
    """Convert GSM8K sample to instruction-response format with CoT."""
    instruction = (
        f"Solve the following math problem step by step.\n\n"
        f"Problem: {sample['question']}"
    )

    final_answer = None
    if "####" in sample["answer"]:
        final_answer = sample["answer"].split("####")[-1].strip()

    return {
        "instruction": instruction,
        "response": sample["answer"],
        "final_answer": final_answer,
    }


def prepare_dataset(subset_size: int = 500, output_dir: str = None):
    """Download GSM8K, format to instruction-response pairs, and save."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading GSM8K dataset from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main")
    print(f"Dataset splits: {ds}")

    train_samples = [format_sample_to_instruction(s) for s in ds["train"]]
    test_samples = [format_sample_to_instruction(s) for s in ds["test"]]

    print(f"Formatted training samples: {len(train_samples)}")
    print(f"Formatted test samples: {len(test_samples)}")

    subset = train_samples[:subset_size]

    # Save full training set
    full_path = os.path.join(output_dir, "gsm8k_formatted_full.json")
    with open(full_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"Saved full training set: {full_path} ({len(train_samples)} samples)")

    # Save training subset
    subset_path = os.path.join(output_dir, "gsm8k_formatted_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved training subset: {subset_path} ({len(subset)} samples)")

    # Save test set for evaluation
    test_path = os.path.join(output_dir, "gsm8k_test.json")
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"Saved test set: {test_path} ({len(test_samples)} samples)")

    print(f"\n{'=' * 60}")
    print("FORMATTED EXAMPLE:")
    print(f"{'=' * 60}")
    print(f"\n[INSTRUCTION]:\n{subset[0]['instruction']}")
    print(f"\n[RESPONSE]:\n{subset[0]['response']}")
    print(f"\n[FINAL ANSWER]: {subset[0]['final_answer']}")

    return train_samples, subset


if __name__ == "__main__":
    prepare_dataset()
