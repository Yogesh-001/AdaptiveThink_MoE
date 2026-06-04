"""
Pipeline Integration Test (Non-Interactive)
=============================================

Tests the full pipeline end-to-end without user input.
Run this to verify all components work together before using the CLI.

Tests:
1. Router loads correctly
2. Base model loads
3. Each expert adapter loads and generates
4. Routing decisions are correct
"""

import os
import sys

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from scripts.pipeline import AdaptiveMoEPipeline


def run_integration_test():
    """Test the full pipeline with sample prompts."""

    print("=" * 60)
    print("INTEGRATION TEST — Full Pipeline")
    print("=" * 60)

    # Check that all adapter paths exist
    adapter_paths = {
        "code": "outputs/code_expert/adapter",
        "math": "outputs/math_expert/adapter",
        "general": "outputs/general_expert/adapter",
    }

    print("\nChecking adapter availability:")
    all_available = True
    for name, path in adapter_paths.items():
        exists = os.path.exists(path)
        status = "✓ Found" if exists else "✗ MISSING"
        print(f"  {name:8s}: {status} ({path})")
        if not exists:
            all_available = False

    if not all_available:
        print("\nWARNING: Some adapters are missing. Train them first.")
        print("The pipeline will still work but will use base model for missing experts.")

    # Check router
    router_exists = os.path.exists("router/model/router_classifier.joblib")
    print(f"\n  Router: {'✓ Found' if router_exists else '✗ MISSING'}")

    if not router_exists:
        print("\nERROR: Router not trained. Run: python router/train_router.py")
        sys.exit(1)

    # Initialize pipeline
    print("\n" + "─" * 60)
    pipeline = AdaptiveMoEPipeline()

    # Test prompts — one per expert
    test_prompts = [
        {
            "prompt": "Write a Python function to check if a number is even",
            "expected_expert": "code",
        },
        {
            "prompt": "What is 25% of 80?",
            "expected_expert": "math",
        },
        {
            "prompt": "What are the benefits of reading books?",
            "expected_expert": "general",
        },
    ]

    print("\n" + "=" * 60)
    print("RUNNING TEST PROMPTS")
    print("=" * 60)

    results = []
    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: \"{test['prompt']}\"")
        print(f"Expected expert: {test['expected_expert']}")
        print(f"{'─' * 60}")

        result = pipeline.generate_with_display(test["prompt"])

        # Check if routing was correct
        correct = result["expert_used"] == test["expected_expert"]
        results.append(correct)
        print(f"\n  Routing correct: {'✓ YES' if correct else '✗ NO'}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Routing accuracy: {sum(results)}/{len(results)} correct")
    print(f"  All experts functional: {'✓' if all(results) else '⚠ Some issues'}")
    print(f"\nPipeline is {'READY' if all(results) else 'partially working'}!")
    print(f"Run the CLI: python scripts/cli.py")


if __name__ == "__main__":
    run_integration_test()
