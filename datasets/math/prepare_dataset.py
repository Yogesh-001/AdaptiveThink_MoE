"""
GSM8K Dataset Preparation for Math Expert Training
====================================================

This script downloads GSM8K (Grade School Math 8K), converts it
into instruction-response format suitable for fine-tuning a math expert.

GSM8K dataset structure (each sample):
- question: A grade-school math word problem
- answer: Step-by-step solution ending with "#### <final_number>"

Why GSM8K?
----------
- It's the standard benchmark for math reasoning in LLMs
- Problems require multi-step reasoning (not just single operations)
- Answers include chain-of-thought (step-by-step), which teaches
  the model HOW to reason, not just give final answers
- Paper: "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)

Chain-of-Thought (CoT) approach:
---------------------------------
GSM8K answers naturally contain step-by-step reasoning:
  "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast..."
  "Step 1: Eggs remaining = 16 - 3 = 13"
  "Step 2: ..."
  "#### 9"

By training on these step-by-step solutions, the model learns to:
1. Break problems into smaller steps
2. Show intermediate calculations
3. Arrive at the correct final answer

This is based on the Chain-of-Thought prompting paper (Wei et al., 2022)
which showed that LLMs reason much better when they "show their work".
"""

import json
import os
from datasets import load_dataset


def format_sample_to_instruction(sample: dict) -> dict:
    """
    Convert a single GSM8K sample into instruction-response format.

    GSM8K answer format:
    - Contains step-by-step reasoning as plain text
    - Ends with "#### <number>" as the final answer marker

    We keep the FULL step-by-step answer (not just the final number) because:
    1. The model learns to reason, not just guess
    2. Chain-of-thought improves accuracy significantly
    3. Users can verify the reasoning path

    Parameters
    ----------
    sample : dict
        A single GSM8K sample with keys: question, answer

    Returns
    -------
    dict
        Formatted sample with: instruction, response, final_answer
    """

    # The instruction is the math problem itself
    # We add a system-level hint to "show your work" to reinforce CoT
    instruction = (
        f"Solve the following math problem step by step.\n\n"
        f"Problem: {sample['question']}"
    )

    # The response is the full chain-of-thought answer
    # GSM8K answers already contain step-by-step reasoning
    response = sample["answer"]

    # Extract the final numerical answer (after "####")
    # This is useful for evaluation later (comparing final numbers)
    final_answer = None
    if "####" in sample["answer"]:
        final_answer = sample["answer"].split("####")[-1].strip()

    return {
        "instruction": instruction,
        "response": response,
        "final_answer": final_answer,
    }


def prepare_dataset(subset_size: int = 500, output_dir: str = None):
    """
    Download GSM8K, format it, and save subsets.

    GSM8K has:
    - train: 7,473 samples (we'll use these for training)
    - test: 1,319 samples (we'll use some for evaluation)

    We start with 500 samples (same approach as code expert) to:
    - Verify the pipeline works
    - Quick iteration cycle
    - Then scale to full dataset if results are good

    Parameters
    ----------
    subset_size : int
        Number of samples for initial experiment
    output_dir : str
        Where to save the formatted data
    """

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading GSM8K dataset from HuggingFace...")
    # GSM8K is hosted under "openai/gsm8k" on HuggingFace
    # split "main" contains train/test
    ds = load_dataset("openai/gsm8k", "main")

    print(f"\nDataset splits: {ds}")
    print(f"\nSample keys: {list(ds['train'][0].keys())}")
    print(f"\nExample raw sample:")
    print(f"  question: {ds['train'][0]['question'][:200]}...")
    print(f"  answer: {ds['train'][0]['answer'][:200]}...")

    # Format training data
    train_samples = []
    for sample in ds["train"]:
        formatted = format_sample_to_instruction(sample)
        train_samples.append(formatted)

    # Format test data (for evaluation later)
    test_samples = []
    for sample in ds["test"]:
        formatted = format_sample_to_instruction(sample)
        test_samples.append(formatted)

    print(f"\nFormatted training samples: {len(train_samples)}")
    print(f"Formatted test samples: {len(test_samples)}")

    # Create subset for initial training
    subset = train_samples[:subset_size]

    # Save full training dataset
    full_path = os.path.join(output_dir, "gsm8k_formatted_full.json")
    with open(full_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"\nSaved full training set: {full_path} ({len(train_samples)} samples)")

    # Save training subset
    subset_path = os.path.join(output_dir, "gsm8k_formatted_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved training subset: {subset_path} ({len(subset)} samples)")

    # Save test set (for evaluation in Phase 6)
    test_path = os.path.join(output_dir, "gsm8k_test.json")
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"Saved test set: {test_path} ({len(test_samples)} samples)")

    # Preview a formatted example
    print("\n" + "=" * 60)
    print("FORMATTED EXAMPLE:")
    print("=" * 60)
    print(f"\n[INSTRUCTION]:\n{subset[0]['instruction']}")
    print(f"\n[RESPONSE]:\n{subset[0]['response']}")
    print(f"\n[FINAL ANSWER]: {subset[0]['final_answer']}")

    return train_samples, subset


if __name__ == "__main__":
    prepare_dataset()
