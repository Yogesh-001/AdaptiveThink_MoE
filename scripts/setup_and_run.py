"""One-click setup: auto-detects state, trains what's needed, runs demo."""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)


def check_adapters_exist() -> dict:
    adapters = {
        "code": "outputs/code_expert/adapter",
        "math": "outputs/math_expert/adapter",
        "general": "outputs/general_expert/adapter",
    }
    return {name: os.path.exists(os.path.join(path, "adapter_config.json"))
            for name, path in adapters.items()}


def check_router_exists() -> bool:
    return os.path.exists("router/model/router_classifier.joblib")


def check_datasets_exist() -> dict:
    datasets = {
        "code": "datasets/code/mbpp_formatted_subset.json",
        "math": "datasets/math/gsm8k_formatted_subset.json",
        "general": "datasets/general/alpaca_formatted_subset.json",
    }
    return {name: os.path.exists(path) for name, path in datasets.items()}


def prepare_all_datasets():
    print(f"\n{'=' * 60}")
    print("STEP 1: PREPARING DATASETS")
    print(f"{'=' * 60}")

    status = check_datasets_exist()

    if not status["code"]:
        print("\n[1/3] Preparing Code dataset (MBPP)...")
        from datasets.code.prepare_dataset import prepare_dataset as prep_code
        prep_code()
    else:
        print("\n[1/3] Code dataset ✓")

    if not status["math"]:
        print("\n[2/3] Preparing Math dataset (GSM8K)...")
        from datasets.math.prepare_dataset import prepare_dataset as prep_math
        prep_math()
    else:
        print("\n[2/3] Math dataset ✓")

    if not status["general"]:
        print("\n[3/3] Preparing General dataset (Alpaca)...")
        from datasets.general.prepare_dataset import prepare_dataset as prep_general
        prep_general()
    else:
        print("\n[3/3] General dataset ✓")


def train_all_experts():
    print(f"\n{'=' * 60}")
    print("STEP 2: TRAINING EXPERT ADAPTERS")
    print(f"{'=' * 60}")

    status = check_adapters_exist()

    if not status["code"]:
        print("\n[1/3] Training Code Expert...")
        start = time.time()
        from training.train_code_expert import train_code_expert
        train_code_expert()
        print(f"  Done in {time.time() - start:.0f}s")
    else:
        print("\n[1/3] Code Expert ✓")

    if not status["math"]:
        print("\n[2/3] Training Math Expert...")
        start = time.time()
        from training.train_math_expert import train_math_expert
        train_math_expert()
        print(f"  Done in {time.time() - start:.0f}s")
    else:
        print("\n[2/3] Math Expert ✓")

    if not status["general"]:
        print("\n[3/3] Training General Expert...")
        start = time.time()
        from training.train_general_expert import train_general_expert
        train_general_expert()
        print(f"  Done in {time.time() - start:.0f}s")
    else:
        print("\n[3/3] General Expert ✓")


def train_router():
    print(f"\n{'=' * 60}")
    print("STEP 3: TRAINING ROUTER")
    print(f"{'=' * 60}")

    if check_router_exists():
        print("\n  Router ✓")
        return

    if not os.path.exists("router/router_training_data.json"):
        print("\n  Generating router training data...")
        from router.create_training_data import create_router_training_data
        create_router_training_data()

    print("\n  Training router classifier...")
    from router.train_router import train_router as _train
    _train()


def run_pipeline_demo():
    print(f"\n{'=' * 60}")
    print("STEP 4: PIPELINE DEMO")
    print(f"{'=' * 60}")

    from scripts.pipeline import AdaptiveMoEPipeline
    pipeline = AdaptiveMoEPipeline()

    demo_prompts = [
        "Write a Python function to find the factorial of a number",
        "If a shirt costs $40 and is on sale for 25% off, what's the final price?",
        "Explain what climate change is in simple terms",
    ]

    for prompt in demo_prompts:
        print(f"\n{'─' * 60}")
        print(f"Prompt: \"{prompt}\"")
        pipeline.generate_with_display(prompt)

    print(f"\n{'=' * 60}")
    print("SETUP COMPLETE!")
    print(f"{'=' * 60}")
    print("\n  python scripts/cli.py              — Interactive CLI")
    print("  python scripts/benchmark_comparison.py  — Run benchmarks")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AdaptiveThink-MoE: Auto Setup & Run               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    adapter_status = check_adapters_exist()
    router_status = check_router_exists()
    all_trained = all(adapter_status.values()) and router_status

    print(f"\n  Current State:")
    for name, exists in adapter_status.items():
        print(f"    {name:8s} expert: {'✓' if exists else '✗'}")
    print(f"    {'router':8s}       : {'✓' if router_status else '✗'}")

    if all_trained:
        print(f"\n  All components ready!")
        run_pipeline_demo()
    else:
        total_start = time.time()
        prepare_all_datasets()
        train_all_experts()
        train_router()
        print(f"\n  Total setup time: {(time.time() - total_start)/60:.1f} min")
        run_pipeline_demo()


if __name__ == "__main__":
    main()
