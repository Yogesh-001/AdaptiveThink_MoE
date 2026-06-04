"""
AdaptiveThink-MoE: Auto Setup & Run
=====================================

ONE-CLICK script for Colab/fresh clones. Handles everything:
1. Checks if trained adapters exist (outputs/ folder)
2. If NOT → prepares data + trains all 3 experts + trains router
3. If YES → skips training, uses existing adapters
4. Runs the full pipeline

Usage on Colab:
---------------
    !git clone https://github.com/Yogesh-001/AdaptiveThink_MoE.git
    %cd AdaptiveThink_MoE
    !pip install -r requirements.txt
    !python scripts/setup_and_run.py

This makes the project demo-ready in one command.
"""

import os
import sys
import time

# Set working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)


def check_adapters_exist() -> dict:
    """
    Check which expert adapters are already trained.

    Returns dict: {expert_name: bool (exists or not)}
    """
    adapters = {
        "code": "outputs/code_expert/adapter",
        "math": "outputs/math_expert/adapter",
        "general": "outputs/general_expert/adapter",
    }

    status = {}
    for name, path in adapters.items():
        # Check for adapter_config.json as proof of a valid adapter
        config_file = os.path.join(path, "adapter_config.json")
        status[name] = os.path.exists(config_file)

    return status


def check_router_exists() -> bool:
    """Check if the router is trained."""
    return os.path.exists("router/model/router_classifier.joblib")


def check_datasets_exist() -> dict:
    """Check which datasets are prepared."""
    datasets = {
        "code": "datasets/code/mbpp_formatted_subset.json",
        "math": "datasets/math/gsm8k_formatted_subset.json",
        "general": "datasets/general/alpaca_formatted_subset.json",
    }
    return {name: os.path.exists(path) for name, path in datasets.items()}


def prepare_all_datasets():
    """Prepare all datasets if not already done."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARING DATASETS")
    print("=" * 60)

    dataset_status = check_datasets_exist()

    if not dataset_status["code"]:
        print("\n[1/3] Preparing Code dataset (MBPP)...")
        from datasets.code.prepare_dataset import prepare_dataset as prepare_code
        prepare_code()
    else:
        print("\n[1/3] Code dataset already exists ✓")

    if not dataset_status["math"]:
        print("\n[2/3] Preparing Math dataset (GSM8K)...")
        from datasets.math.prepare_dataset import prepare_dataset as prepare_math
        prepare_math()
    else:
        print("\n[2/3] Math dataset already exists ✓")

    if not dataset_status["general"]:
        print("\n[3/3] Preparing General dataset (Alpaca)...")
        from datasets.general.prepare_dataset import prepare_dataset as prepare_general
        prepare_general()
    else:
        print("\n[3/3] General dataset already exists ✓")


def train_all_experts():
    """Train all expert adapters."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING EXPERT ADAPTERS")
    print("=" * 60)

    adapter_status = check_adapters_exist()

    if not adapter_status["code"]:
        print("\n[1/3] Training Code Expert...")
        start = time.time()
        from training.train_code_expert import train_code_expert
        train_code_expert()
        print(f"  Code Expert trained in {time.time() - start:.0f}s")
    else:
        print("\n[1/3] Code Expert already trained ✓")

    if not adapter_status["math"]:
        print("\n[2/3] Training Math Expert...")
        start = time.time()
        from training.train_math_expert import train_math_expert
        train_math_expert()
        print(f"  Math Expert trained in {time.time() - start:.0f}s")
    else:
        print("\n[2/3] Math Expert already trained ✓")

    if not adapter_status["general"]:
        print("\n[3/3] Training General Expert...")
        start = time.time()
        from training.train_general_expert import train_general_expert
        train_general_expert()
        print(f"  General Expert trained in {time.time() - start:.0f}s")
    else:
        print("\n[3/3] General Expert already trained ✓")


def train_router():
    """Train the router if not already done."""
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING ROUTER")
    print("=" * 60)

    if check_router_exists():
        print("\n  Router already trained ✓")
        return

    # Generate router training data if needed
    if not os.path.exists("router/router_training_data.json"):
        print("\n  Generating router training data...")
        from router.create_training_data import create_router_training_data
        create_router_training_data()

    # Train the router
    print("\n  Training router classifier...")
    from router.train_router import train_router as _train_router
    _train_router()


def run_pipeline_demo():
    """Run a quick demo of the full pipeline."""
    print("\n" + "=" * 60)
    print("STEP 4: RUNNING PIPELINE DEMO")
    print("=" * 60)

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
        result = pipeline.generate_with_display(prompt)

    print(f"\n{'=' * 60}")
    print("SETUP COMPLETE — All systems operational!")
    print(f"{'=' * 60}")
    print("\nYou can now:")
    print("  • Run the CLI:        python scripts/cli.py")
    print("  • Run benchmarks:     python scripts/benchmark_comparison.py")
    print("  • Run integration:    python scripts/test_pipeline.py")


def main():
    """Main entry point — auto-detects state and runs what's needed."""

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AdaptiveThink-MoE: Auto Setup & Run               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Check current state
    adapter_status = check_adapters_exist()
    router_status = check_router_exists()
    all_trained = all(adapter_status.values()) and router_status

    print(f"\n  Current State:")
    print(f"  {'─' * 40}")
    for name, exists in adapter_status.items():
        print(f"    {name:8s} expert: {'✓ trained' if exists else '✗ needs training'}")
    print(f"    {'router':8s}       : {'✓ trained' if router_status else '✗ needs training'}")

    if all_trained:
        print(f"\n  All components ready! Skipping training...")
        print(f"  Running demo directly...")
        run_pipeline_demo()
    else:
        total_start = time.time()
        print(f"\n  Some components need training. Starting full setup...")

        # Step 1: Prepare datasets
        prepare_all_datasets()

        # Step 2: Train experts
        train_all_experts()

        # Step 3: Train router
        train_router()

        total_time = time.time() - total_start
        print(f"\n  Total setup time: {total_time/60:.1f} minutes")

        # Step 4: Demo
        run_pipeline_demo()


if __name__ == "__main__":
    main()
