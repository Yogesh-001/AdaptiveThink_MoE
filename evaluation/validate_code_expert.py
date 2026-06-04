"""
Code Expert Validation Script
==============================

After training the LoRA adapter, we need to verify it actually works.
This script loads the base model + trained LoRA adapter and tests it
on coding prompts to see if the expert generates good Python code.

Why validate separately?
------------------------
1. Training loss going down ≠ model generating good code
   (it might memorize but not generalize)
2. We need to SEE actual outputs to assess quality
3. This is qualitative evaluation — we'll do quantitative benchmarks
   in Phase 6 (HumanEval)

How LoRA loading works at inference:
------------------------------------
1. Load the base model (same as before, frozen)
2. Load LoRA adapter on top (merges small matrices into the model)
3. The model now behaves as if it was fine-tuned, but we only stored
   the tiny LoRA weights

This is the foundation of our MoE system:
- Same base model loaded once
- Different LoRA adapters for different skills
- Router decides which adapter to activate
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_expert_model(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "outputs/code_expert/adapter",
):
    """
    Load base model + LoRA adapter.

    This two-step loading is the core pattern for our MoE system:
    1. Load base model (expensive, done once)
    2. Load LoRA adapter (cheap, can swap between experts)

    Parameters
    ----------
    base_model_name : str
        HuggingFace model ID for the base model
    adapter_path : str
        Path to the saved LoRA adapter directory

    Returns
    -------
    model, tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the base model (same as training)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # Now load the LoRA adapter on top
    # PeftModel.from_pretrained merges the adapter into the base model
    # The adapter is tiny (~5-20MB) compared to the base model (~1GB)
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Set to evaluation mode (disables dropout, batch norm behaves differently)
    model.eval()

    print(f"Model loaded on: {device}")
    return model, tokenizer


def generate_code(model, tokenizer, problem: str, max_tokens: int = 256) -> str:
    """
    Generate code for a given problem using the code expert.

    Parameters
    ----------
    model : The model with LoRA adapter loaded
    tokenizer : The tokenizer
    problem : str
        Natural language description of the coding problem
    max_tokens : int
        Maximum tokens to generate

    Returns
    -------
    str : Generated code
    """

    # Format the prompt exactly as we trained (instruction format)
    # This is critical: the model learned this specific template
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": problem},
    ]

    # apply_chat_template: Converts messages to the model's expected format
    # add_generation_prompt=True: Adds the start of assistant turn
    # so the model knows it should start generating the response
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # <|im_start|>assistant\n
    )

    # Tokenize and move to model's device
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate with controlled randomness
    with torch.no_grad():  # Disable gradient computation (saves memory during inference)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,

            # Temperature: Controls randomness
            # 0.2 = mostly deterministic (good for code — we want correctness)
            # 1.0 = very random, 0.0 = greedy (always pick highest probability)
            temperature=0.2,

            # top_p (nucleus sampling): Only sample from top tokens whose
            # cumulative probability reaches p
            # 0.9 = consider top 90% probability mass
            # This prevents the model from picking very unlikely tokens
            top_p=0.9,

            # do_sample=True: Actually sample from the distribution
            # If False, always picks the highest probability token (greedy)
            do_sample=True,

            # repetition_penalty: Discourages repeating the same text
            # 1.1 = mild penalty (1.0 = no penalty)
            repetition_penalty=1.1,
        )

    # Decode only the NEW tokens (not the input prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def run_validation():
    """
    Test the code expert on various problems to assess quality.

    We test on problems NOT in the training set to check generalization.
    These problems range from easy to medium difficulty.
    """

    # Test prompts (deliberately different from MBPP training data)
    test_problems = [
        # Easy: basic function
        "Write a Python function to find the maximum element in a list.",

        # Medium: requires understanding of algorithm
        "Write a Python function to check if a string is a palindrome.",

        # Medium: requires list manipulation
        "Write a Python function that takes a list of numbers and returns "
        "a new list with only the even numbers, sorted in ascending order.",

        # Harder: requires understanding of data structures
        "Write a Python function to find the two numbers in a list that add up to a target sum. "
        "Return their indices.",
    ]

    # Load the trained expert
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/code_expert/adapter"

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at '{adapter_path}'")
        print("Have you trained the code expert yet?")
        print("Run: python training/train_code_expert.py")
        sys.exit(1)

    model, tokenizer = load_expert_model(adapter_path=adapter_path)

    # Generate and display results
    print("\n" + "=" * 70)
    print("CODE EXPERT VALIDATION")
    print("=" * 70)

    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'─' * 70}")
        print(f"Problem {i}: {problem}")
        print(f"{'─' * 70}")

        response = generate_code(model, tokenizer, problem)
        print(f"\nGenerated:\n{response}")

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nManual assessment checklist:")
    print("  [ ] Does the code run without syntax errors?")
    print("  [ ] Does it solve the stated problem?")
    print("  [ ] Is the code clean and idiomatic Python?")
    print("  [ ] Does it handle edge cases?")


if __name__ == "__main__":
    run_validation()
