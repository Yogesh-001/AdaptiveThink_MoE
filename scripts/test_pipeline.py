"""Pipeline integration test — verifies all components work together."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from scripts.pipeline import AdaptiveMoEPipeline


def run_integration_test():
    """Test the full pipeline with sample prompts."""
    print("=" * 60)
    print("INTEGRATION TEST — Full Pipeline")
    print("=" * 60)

    # Check adapters
    adapter_paths = {
        "code": "outputs/code_expert/adapter",
        "math": "outputs/math_expert/adapter",
        "general": "outputs/general_expert/adapter",
    }

    print("\nChecking adapters:")
    for name, path in adapter_paths.items():
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {name:8s}: {status}")

    router_exists = os.path.exists("router/model/router_classifier.joblib")
    print(f"  {'router':8s}: {'✓' if router_exists else '✗ MISSING'}")

    if not router_exists:
        print("\nERROR: Router not trained. Run: python router/train_router.py")
        sys.exit(1)

    # Initialize pipeline
    print(f"\n{'─' * 60}")
    pipeline = AdaptiveMoEPipeline()

    # Test prompts
    test_prompts = [
        {"prompt": "Write a Python function to check if a number is even", "expected_expert": "code"},
        {"prompt": "What is 25% of 80?", "expected_expert": "math"},
        {"prompt": "What are the benefits of reading books?", "expected_expert": "general"},
    ]

    print(f"\n{'=' * 60}")
    print("RUNNING TEST PROMPTS")
    print(f"{'=' * 60}")

    results = []
    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: \"{test['prompt']}\"")
        print(f"Expected: {test['expected_expert']}")
        print(f"{'─' * 60}")

        result = pipeline.generate_with_display(test["prompt"])
        correct = result["expert_used"] == test["expected_expert"]
        results.append(correct)
        print(f"\n  Routing correct: {'✓' if correct else '✗'}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {sum(results)}/{len(results)} routing correct")
    print(f"Pipeline is {'READY' if all(results) else 'partially working'}!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_integration_test()
