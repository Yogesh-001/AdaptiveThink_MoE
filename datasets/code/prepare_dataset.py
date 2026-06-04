"""
MBPP Dataset Preparation for Code Expert Training
==================================================

This script downloads MBPP (Mostly Basic Python Problems), converts it
into instruction-response format suitable for fine-tuning a language model,
and saves a 500-sample subset for initial experiments.

MBPP dataset structure (each sample):
- text: Natural language description of the problem
- code: Python solution
- test_list: List of assert statements to verify the solution
- task_id: Unique identifier

We convert this into instruction-response pairs because:
- LLMs learn best from (instruction, response) format during fine-tuning
- This matches how users will interact: "Write a function to..."
- It teaches the model to follow instructions, not just complete code

The technique of instruction-tuning comes from the FLAN paper (Google, 2022)
and InstructGPT (OpenAI, 2022). For code specifically, we follow patterns
similar to Code Alpaca and WizardCoder.
"""

import json
import os
from datasets import load_dataset


def format_sample_to_instruction(sample: dict) -> dict:
    """
    Convert a single MBPP sample into instruction-response format.

    Why this format?
    ----------------
    Modern LLMs are fine-tuned with a chat/instruction template:
      - "instruction": What the user asks
      - "response": What the model should generate

    For code tasks, we include test cases in the instruction because:
    1. It gives the model clarity on expected behavior (like TDD)
    2. It improves code correctness during generation
    3. Research shows providing examples/tests in prompts improves code quality
       (ref: "Self-Debugging" paper, Chen et al. 2023)

    Parameters
    ----------
    sample : dict
        A single MBPP sample with keys: text, code, test_list, task_id

    Returns
    -------
    dict
        Formatted sample with: instruction, response, task_id
    """

    # Build the instruction from the problem description + test cases
    # We include test cases so the model learns what "correct" means
    test_cases = "\n".join(sample["test_list"])

    instruction = (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {sample['text']}\n\n"
        f"Test cases:\n{test_cases}"
    )

    # The response is simply the ground-truth code solution
    response = sample["code"]

    return {
        "instruction": instruction,
        "response": response,
        "task_id": sample["task_id"],
    }


def prepare_dataset(subset_size: int = 500, output_dir: str = None):
    """
    Download MBPP, format it, and save a subset.

    Why 500 samples first?
    ----------------------
    - Fast iteration: Training on 500 samples takes minutes, not hours
    - Verify pipeline: Catch bugs in formatting/training before full run
    - Validate quality: Check if the model learns the pattern at all
    - This follows the "start small, scale up" principle common in ML research

    Parameters
    ----------
    subset_size : int
        Number of samples for the initial experiment (default: 500)
    output_dir : str
        Where to save the formatted data
    """

    if output_dir is None:
        # Save next to this script
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading MBPP dataset from HuggingFace...")
    # load_dataset downloads from HuggingFace Hub and caches locally
    # MBPP has splits: train (374), test (500), validation (90), prompt (10)
    # We use the full namespace "google-research-datasets/mbpp" as required by HF Hub
    ds = load_dataset("google-research-datasets/mbpp")

    print(f"\nDataset splits: {ds}")
    print(f"\nSample keys: {list(ds['train'][0].keys())}")
    print(f"\nExample raw sample:")
    print(f"  text: {ds['train'][0]['text']}")
    print(f"  code: {ds['train'][0]['code'][:100]}...")
    print(f"  test_list: {ds['train'][0]['test_list']}")

    # Combine train + test + validation for more data
    # MBPP is small (~974 total), so we use everything for training the expert
    # In research, this is acceptable when your evaluation is on a DIFFERENT
    # benchmark (we'll evaluate on HumanEval in Phase 6)
    all_samples = []
    for split in ["train", "test", "validation"]:
        for sample in ds[split]:
            formatted = format_sample_to_instruction(sample)
            all_samples.append(formatted)

    print(f"\nTotal formatted samples: {len(all_samples)}")

    # Create the subset (first N samples)
    subset = all_samples[:subset_size]

    # Save full dataset
    full_path = os.path.join(output_dir, "mbpp_formatted_full.json")
    with open(full_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Saved full dataset: {full_path} ({len(all_samples)} samples)")

    # Save subset for initial experiments
    subset_path = os.path.join(output_dir, "mbpp_formatted_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved subset: {subset_path} ({len(subset)} samples)")

    # Preview a formatted example
    print("\n" + "=" * 60)
    print("FORMATTED EXAMPLE:")
    print("=" * 60)
    print(f"\n[INSTRUCTION]:\n{subset[0]['instruction']}")
    print(f"\n[RESPONSE]:\n{subset[0]['response']}")

    return all_samples, subset


if __name__ == "__main__":
    prepare_dataset()
