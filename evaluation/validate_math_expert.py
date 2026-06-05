"""Validate Math Expert on test problems and check numerical accuracy."""

import sys
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_expert_model(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "outputs/math_expert/adapter",
):
    """Load base model with math LoRA adapter."""
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


def generate_solution(model, tokenizer, problem: str, max_tokens: int = 300) -> str:
    """Generate a step-by-step math solution."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. Solve problems step by step, "
                "showing your reasoning clearly. End with the final numerical answer."
            ),
        },
        {"role": "user", "content": problem},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def extract_final_answer(text: str) -> str:
    """Extract the final numerical answer from a generated solution."""
    # Pattern: "#### number"
    match = re.search(r"####\s*([\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern: "the answer is X" or "= X"
    match = re.search(r"(?:the answer is|answer:|=)\s*([\d,]+\.?\d*)\s*$", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def run_validation(use_test_set: bool = False):
    """Validate the math expert on problems."""
    adapter_path = "outputs/math_expert/adapter"
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            adapter_path = arg
            break

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at '{adapter_path}'")
        print("Run: python training/train_math_expert.py")
        sys.exit(1)

    model, tokenizer = load_expert_model(adapter_path=adapter_path)

    if use_test_set and os.path.exists("datasets/math/gsm8k_test.json"):
        print(f"\n{'=' * 70}")
        print("MATH EXPERT - GSM8K TEST EVALUATION")
        print(f"{'=' * 70}")

        with open("datasets/math/gsm8k_test.json", "r") as f:
            test_data = json.load(f)

        num_eval = 20
        correct = 0

        for i, sample in enumerate(test_data[:num_eval]):
            response = generate_solution(model, tokenizer, sample["instruction"])
            predicted = extract_final_answer(response)
            is_correct = predicted == sample["final_answer"]
            correct += int(is_correct)
            status = "✓" if is_correct else "✗"
            print(f"  [{status}] Problem {i+1}: expected={sample['final_answer']}, got={predicted}")

        print(f"\nAccuracy: {correct}/{num_eval} = {correct/num_eval*100:.1f}%")

    else:
        test_problems = [
            {
                "problem": "Solve the following math problem step by step.\n\nProblem: A store sells apples for $2 each and oranges for $3 each. If Sarah buys 4 apples and 5 oranges, how much does she spend in total?",
                "expected": "23",
            },
            {
                "problem": "Solve the following math problem step by step.\n\nProblem: A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes?",
                "expected": "150",
            },
            {
                "problem": "Solve the following math problem step by step.\n\nProblem: If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?",
                "expected": "40",
            },
            {
                "problem": "Solve the following math problem step by step.\n\nProblem: Tom has 48 marbles. He gives 1/4 of them to his friend and then buys 12 more. How many marbles does he have now?",
                "expected": "48",
            },
        ]

        print(f"\n{'=' * 70}")
        print("MATH EXPERT VALIDATION")
        print(f"{'=' * 70}")

        correct = 0
        total = len(test_problems)

        for i, test in enumerate(test_problems, 1):
            print(f"\n{'─' * 70}")
            print(f"Problem {i}: {test['problem']}")
            print(f"Expected: {test['expected']}")
            print(f"{'─' * 70}")

            response = generate_solution(model, tokenizer, test["problem"])
            predicted = extract_final_answer(response)
            is_correct = predicted == test["expected"]
            correct += int(is_correct)

            print(f"\nGenerated:\n{response}")
            print(f"\nExtracted: {predicted} | Correct: {'✓' if is_correct else '✗'}")

        print(f"\n{'=' * 70}")
        print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    use_test_set = "--test" in sys.argv
    run_validation(use_test_set=use_test_set)
