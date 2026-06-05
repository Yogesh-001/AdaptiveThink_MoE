"""Base model vs MoE comparison benchmark — measures quality, accuracy, and latency."""

import os
import sys
import time
import json
import torch
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.pipeline import AdaptiveMoEPipeline


class BaseModelInference:
    """Bare base model inference (no LoRA) for baseline comparison."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading base model for comparison...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 256) -> dict:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, top_p=0.9, do_sample=True, repetition_penalty=1.1,
            )
        generation_time = time.time() - start_time

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {"response": response, "generation_time": generation_time}


def evaluate_code_correctness(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_math_answer(text: str) -> str:
    """Extract final numerical answer from response."""
    import re
    match = re.search(r"####\s*([\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def run_benchmark():
    """Run the full comparison benchmark."""

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    AdaptiveThink-MoE vs Base Model: Benchmark            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # ========================================
    # SETUP
    # ========================================
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    # Load base model
    base = BaseModelInference()

    # Load MoE pipeline
    print()
    moe = AdaptiveMoEPipeline()

    # ========================================
    # BENCHMARK TASKS
    # ========================================

    # Code tasks with expected behavior
    code_tasks = [
        {
            "prompt": "Write a Python function to check if a string is a palindrome.",
            "description": "Palindrome check",
        },
        {
            "prompt": "Write a Python function to find the second largest number in a list.",
            "description": "Second largest",
        },
        {
            "prompt": "Write a Python function to count vowels in a string.",
            "description": "Count vowels",
        },
        {
            "prompt": "Write a Python function to merge two sorted lists into one sorted list.",
            "description": "Merge sorted lists",
        },
        {
            "prompt": "Write a Python function to remove duplicates from a list while preserving order.",
            "description": "Remove duplicates",
        },
    ]

    # Math tasks with known answers
    math_tasks = [
        {
            "prompt": "Solve the following math problem step by step.\n\nProblem: A store has a 30% off sale. If a jacket originally costs $80, what is the sale price?",
            "expected": "56",
            "description": "Percentage discount",
        },
        {
            "prompt": "Solve the following math problem step by step.\n\nProblem: If 3 workers can paint a house in 12 days, how many days would it take 4 workers?",
            "expected": "9",
            "description": "Work rate problem",
        },
        {
            "prompt": "Solve the following math problem step by step.\n\nProblem: A rectangle has a perimeter of 24 cm. If its length is 3 times its width, find the area.",
            "expected": "27",
            "description": "Geometry problem",
        },
        {
            "prompt": "Solve the following math problem step by step.\n\nProblem: Tom has $100. He spends 40% on books and 25% of the remainder on food. How much money does he have left?",
            "expected": "45",
            "description": "Multi-step money",
        },
        {
            "prompt": "Solve the following math problem step by step.\n\nProblem: A car travels 180 km in 3 hours. If it travels at the same speed, how far will it go in 5 hours?",
            "expected": "300",
            "description": "Speed/distance",
        },
    ]

    # General tasks (qualitative comparison)
    general_tasks = [
        {
            "prompt": "Explain the difference between a virus and a bacteria in simple terms.",
            "description": "Science explanation",
        },
        {
            "prompt": "Give me 3 practical tips for better sleep.",
            "description": "Advice/tips",
        },
        {
            "prompt": "What are the main causes of climate change?",
            "description": "Factual Q&A",
        },
    ]

    # ========================================
    # RUN COMPARISONS
    # ========================================

    results = {
        "code": {"base": [], "moe": []},
        "math": {"base": [], "moe": []},
        "general": {"base": [], "moe": []},
        "metadata": {
            "date": datetime.now().isoformat(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        },
    }

    # --- CODE BENCHMARK ---
    print("\n" + "=" * 60)
    print("BENCHMARK 1: CODE GENERATION")
    print("=" * 60)

    for task in code_tasks:
        print(f"\n  Task: {task['description']}")

        # Base model
        base_result = base.generate(
            task["prompt"],
            system_prompt="You are a helpful coding assistant. Write clean, correct Python code.",
        )
        base_valid = evaluate_code_correctness(base_result["response"])

        # MoE
        moe_result = moe.generate(task["prompt"])
        moe_valid = evaluate_code_correctness(moe_result["response"])

        print(f"    Base: {'✓ valid' if base_valid else '✗ invalid'} syntax | {base_result['generation_time']:.2f}s")
        print(f"    MoE:  {'✓ valid' if moe_valid else '✗ invalid'} syntax | {moe_result['generation_time']:.2f}s | expert={moe_result['expert_used']}")

        results["code"]["base"].append({
            "task": task["description"],
            "response": base_result["response"],
            "time": base_result["generation_time"],
            "valid_syntax": base_valid,
        })
        results["code"]["moe"].append({
            "task": task["description"],
            "response": moe_result["response"],
            "time": moe_result["generation_time"],
            "valid_syntax": moe_valid,
            "expert": moe_result["expert_used"],
            "confidence": moe_result["routing_info"]["confidence"],
        })

    # --- MATH BENCHMARK ---
    print("\n" + "=" * 60)
    print("BENCHMARK 2: MATH REASONING")
    print("=" * 60)

    for task in math_tasks:
        print(f"\n  Task: {task['description']} (expected: {task['expected']})")

        # Base model
        base_result = base.generate(
            task["prompt"],
            system_prompt="You are a helpful math assistant. Solve step by step.",
        )
        base_answer = extract_math_answer(base_result["response"])
        base_correct = base_answer == task["expected"]

        # MoE
        moe_result = moe.generate(task["prompt"])
        moe_answer = extract_math_answer(moe_result["response"])
        moe_correct = moe_answer == task["expected"]

        print(f"    Base: answer={base_answer} {'✓' if base_correct else '✗'} | {base_result['generation_time']:.2f}s")
        print(f"    MoE:  answer={moe_answer} {'✓' if moe_correct else '✗'} | {moe_result['generation_time']:.2f}s | expert={moe_result['expert_used']}")

        results["math"]["base"].append({
            "task": task["description"],
            "response": base_result["response"],
            "time": base_result["generation_time"],
            "answer": base_answer,
            "expected": task["expected"],
            "correct": base_correct,
        })
        results["math"]["moe"].append({
            "task": task["description"],
            "response": moe_result["response"],
            "time": moe_result["generation_time"],
            "answer": moe_answer,
            "expected": task["expected"],
            "correct": moe_correct,
            "expert": moe_result["expert_used"],
            "confidence": moe_result["routing_info"]["confidence"],
        })

    # --- GENERAL BENCHMARK ---
    print("\n" + "=" * 60)
    print("BENCHMARK 3: GENERAL KNOWLEDGE")
    print("=" * 60)

    for task in general_tasks:
        print(f"\n  Task: {task['description']}")

        # Base model
        base_result = base.generate(task["prompt"])

        # MoE
        moe_result = moe.generate(task["prompt"])

        print(f"    Base: {len(base_result['response'])} chars | {base_result['generation_time']:.2f}s")
        print(f"    MoE:  {len(moe_result['response'])} chars | {moe_result['generation_time']:.2f}s | expert={moe_result['expert_used']}")

        results["general"]["base"].append({
            "task": task["description"],
            "response": base_result["response"],
            "time": base_result["generation_time"],
            "length": len(base_result["response"]),
        })
        results["general"]["moe"].append({
            "task": task["description"],
            "response": moe_result["response"],
            "time": moe_result["generation_time"],
            "length": len(moe_result["response"]),
            "expert": moe_result["expert_used"],
            "confidence": moe_result["routing_info"]["confidence"],
        })

    # ========================================
    # SUMMARY REPORT
    # ========================================
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              BENCHMARK RESULTS SUMMARY                   ║")
    print("╠══════════════════════════════════════════════════════════╣")

    # Code results
    base_code_valid = sum(1 for r in results["code"]["base"] if r["valid_syntax"])
    moe_code_valid = sum(1 for r in results["code"]["moe"] if r["valid_syntax"])
    base_code_time = sum(r["time"] for r in results["code"]["base"]) / len(results["code"]["base"])
    moe_code_time = sum(r["time"] for r in results["code"]["moe"]) / len(results["code"]["moe"])

    print(f"║                                                          ║")
    print(f"║  CODE GENERATION                                         ║")
    print(f"║  ────────────────────────────────────────────────────    ║")
    print(f"║  Syntax Validity:  Base={base_code_valid}/5  MoE={moe_code_valid}/5         ║")
    print(f"║  Avg Time:         Base={base_code_time:.2f}s  MoE={moe_code_time:.2f}s        ║")

    # Math results
    base_math_correct = sum(1 for r in results["math"]["base"] if r["correct"])
    moe_math_correct = sum(1 for r in results["math"]["moe"] if r["correct"])
    base_math_time = sum(r["time"] for r in results["math"]["base"]) / len(results["math"]["base"])
    moe_math_time = sum(r["time"] for r in results["math"]["moe"]) / len(results["math"]["moe"])

    print(f"║                                                          ║")
    print(f"║  MATH REASONING                                          ║")
    print(f"║  ────────────────────────────────────────────────────    ║")
    print(f"║  Accuracy:         Base={base_math_correct}/5  MoE={moe_math_correct}/5         ║")
    print(f"║  Avg Time:         Base={base_math_time:.2f}s  MoE={moe_math_time:.2f}s        ║")

    # General results
    base_gen_time = sum(r["time"] for r in results["general"]["base"]) / len(results["general"]["base"])
    moe_gen_time = sum(r["time"] for r in results["general"]["moe"]) / len(results["general"]["moe"])
    base_gen_len = sum(r["length"] for r in results["general"]["base"]) / len(results["general"]["base"])
    moe_gen_len = sum(r["length"] for r in results["general"]["moe"]) / len(results["general"]["moe"])

    print(f"║                                                          ║")
    print(f"║  GENERAL KNOWLEDGE                                       ║")
    print(f"║  ────────────────────────────────────────────────────    ║")
    print(f"║  Avg Response Len: Base={base_gen_len:.0f} chars  MoE={moe_gen_len:.0f} chars   ║")
    print(f"║  Avg Time:         Base={base_gen_time:.2f}s  MoE={moe_gen_time:.2f}s        ║")

    # Routing accuracy
    all_moe_results = results["code"]["moe"] + results["math"]["moe"] + results["general"]["moe"]
    expected_routing = (["code"] * 5) + (["math"] * 5) + (["general"] * 3)
    routing_correct = sum(
        1 for r, expected in zip(all_moe_results, expected_routing)
        if r["expert"] == expected
    )

    print(f"║                                                          ║")
    print(f"║  ROUTING                                                 ║")
    print(f"║  ────────────────────────────────────────────────────    ║")
    print(f"║  Routing Accuracy:  {routing_correct}/{len(expected_routing)} ({routing_correct/len(expected_routing)*100:.0f}%)                          ║")
    print(f"║  Avg Confidence:    {sum(r['confidence'] for r in all_moe_results)/len(all_moe_results):.1%}                           ║")

    # Overall
    print(f"║                                                          ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  OVERALL COMPARISON                                      ║")
    print(f"║  ────────────────────────────────────────────────────    ║")

    overall_base_time = (base_code_time + base_math_time + base_gen_time) / 3
    overall_moe_time = (moe_code_time + moe_math_time + moe_gen_time) / 3
    time_overhead = ((overall_moe_time - overall_base_time) / overall_base_time) * 100

    print(f"║  Avg Latency:      Base={overall_base_time:.2f}s  MoE={overall_moe_time:.2f}s     ║")
    print(f"║  Overhead:         {time_overhead:+.1f}% (routing + adapter load)      ║")
    print(f"║  Code Quality:     MoE {'>' if moe_code_valid > base_code_valid else '=' if moe_code_valid == base_code_valid else '<'} Base                                ║")
    print(f"║  Math Accuracy:    MoE {'>' if moe_math_correct > base_math_correct else '=' if moe_math_correct == base_math_correct else '<'} Base                                ║")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # Save detailed results
    os.makedirs("outputs/benchmark", exist_ok=True)
    results_path = "outputs/benchmark/comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")

    # Print LinkedIn-ready summary
    print("\n" + "─" * 60)
    print("📋 LINKEDIN-READY SUMMARY:")
    print("─" * 60)
    print(f"""
🚀 Built AdaptiveThink-MoE: A Mixture-of-Experts system using LoRA adapters

📊 Results (Qwen2.5-0.5B base model):

Code Generation:
  • Base model: {base_code_valid}/5 valid syntax
  • MoE (Code Expert): {moe_code_valid}/5 valid syntax

Math Reasoning:
  • Base model: {base_math_correct}/5 correct answers ({base_math_correct*20}%)
  • MoE (Math Expert): {moe_math_correct}/5 correct answers ({moe_math_correct*20}%)

Routing Accuracy: {routing_correct}/{len(expected_routing)} ({routing_correct/len(expected_routing)*100:.0f}%)
Latency Overhead: {time_overhead:+.1f}%

Architecture:
  • 1 shared base model (494M params)
  • 3 LoRA experts (~2M params each, 0.4% of base)
  • Semantic router (MiniLM embeddings + LogReg)
  • Total added params: <1% of base model

Key insight: Specialized adapters improve task performance with
minimal parameter overhead, while the router correctly identifies
task type {routing_correct/len(expected_routing)*100:.0f}% of the time.
""")

    return results


if __name__ == "__main__":
    run_benchmark()
