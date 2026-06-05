"""Validate Code Expert by generating solutions for test problems."""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_expert_model(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "outputs/code_expert/adapter",
):
    """Load base model with code LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print(f"Model loaded on: {device}")
    return model, tokenizer


def generate_code(model, tokenizer, problem: str, max_tokens: int = 256) -> str:
    """Generate code for a given problem."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": problem},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_validation():
    """Test the code expert on various problems."""
    test_problems = [
        "Write a Python function to find the maximum element in a list.",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python function that takes a list of numbers and returns "
        "a new list with only the even numbers, sorted in ascending order.",
        "Write a Python function to find the two numbers in a list that add up to a target sum. "
        "Return their indices.",
    ]

    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/code_expert/adapter"

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at '{adapter_path}'")
        print("Run: python training/train_code_expert.py")
        sys.exit(1)

    model, tokenizer = load_expert_model(adapter_path=adapter_path)

    print(f"\n{'=' * 70}")
    print("CODE EXPERT VALIDATION")
    print(f"{'=' * 70}")

    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'─' * 70}")
        print(f"Problem {i}: {problem}")
        print(f"{'─' * 70}")
        response = generate_code(model, tokenizer, problem)
        print(f"\nGenerated:\n{response}")

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_validation()
