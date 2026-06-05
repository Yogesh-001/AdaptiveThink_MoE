"""AdaptiveThink-MoE Interactive CLI."""

import os
import sys
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from scripts.pipeline import AdaptiveMoEPipeline


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           AdaptiveThink-MoE Interactive CLI              ║")
    print("║  Experts: Code │ Math │ General                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Just type a prompt     → Auto-routes to best expert")
    print("  [code] your prompt     → Force code expert")
    print("  [math] your prompt     → Force math expert")
    print("  [general] your prompt  → Force general expert")
    print("  /route <text>          → Show routing only")
    print("  /experts               → List available experts")
    print("  /quit                  → Exit")
    print()
    print("─" * 60)


def parse_input(user_input: str) -> tuple:
    """Parse for [expert] prefix. Returns (prompt, force_expert or None)."""
    match = re.match(r"\[(code|math|general)\]\s*(.*)", user_input, re.IGNORECASE)
    if match:
        return match.group(2), match.group(1).lower()
    return user_input, None


def main():
    print_banner()

    print("Loading pipeline...\n")
    try:
        pipeline = AdaptiveMoEPipeline()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure adapters and router are trained.")
        sys.exit(1)

    print("\nReady! Type your prompt below.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("\nGoodbye!")
                break

            if user_input.lower() == "/experts":
                print("\n  Available experts:")
                for name, path in pipeline.adapter_paths.items():
                    exists = "✓" if os.path.exists(path) else "✗ (not trained)"
                    print(f"    {name:8s} → {path} {exists}")
                print()
                continue

            if user_input.lower().startswith("/route "):
                text = user_input[7:].strip()
                if text:
                    result = pipeline.router.route(text)
                    print(f"\n  → Expert: {result['expert'].upper()} | Confidence: {result['confidence']:.1%}")
                    for expert, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                        bar = "█" * int(prob * 20)
                        print(f"    {expert:8s}: {prob:.3f} {bar}")
                    print()
                continue

            prompt, force_expert = parse_input(user_input)
            if not prompt:
                continue

            pipeline.generate_with_display(prompt, force_expert=force_expert)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


if __name__ == "__main__":
    main()
