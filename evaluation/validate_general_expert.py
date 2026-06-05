"""Validate General Expert on diverse instruction-following tasks."""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_expert_model(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "outputs/general_expert/adapter",
):
    """Load base model with general LoRA adapter."""
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


def generate_response(model, tokenizer, instruction: str, max_tokens: int = 300) -> str:
    """Generate a response using the general expert."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, knowledgeable assistant. Provide clear, "
                "accurate, and well-organized responses to any question or task."
            ),
        },
        {"role": "user", "content": instruction},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_validation():
    """Test the general expert on diverse tasks."""
    adapter_path = "outputs/general_expert/adapter"
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            adapter_path = arg
            break

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at '{adapter_path}'")
        print("Run: python training/train_general_expert.py")
        sys.exit(1)

    model, tokenizer = load_expert_model(adapter_path=adapter_path)

    test_prompts = [
        {"category": "Explanation", "instruction": "Explain what machine learning is in simple terms that a 10-year-old could understand."},
        {"category": "Comparison", "instruction": "What are the main differences between Python and JavaScript?"},
        {"category": "Advice", "instruction": "Give me 3 tips for improving productivity while working from home."},
        {"category": "Creative", "instruction": "Write a short paragraph describing a sunset over the ocean."},
        {"category": "Factual Q&A", "instruction": "What are the three states of matter? Give a brief explanation of each."},
    ]

    print(f"\n{'=' * 70}")
    print("GENERAL EXPERT VALIDATION")
    print(f"{'=' * 70}")

    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'─' * 70}")
        print(f"[{test['category']}] Problem {i}: {test['instruction']}")
        print(f"{'─' * 70}")
        response = generate_response(model, tokenizer, test["instruction"])
        print(f"\nGenerated:\n{response}")

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_validation()
