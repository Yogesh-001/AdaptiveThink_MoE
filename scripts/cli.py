"""
AdaptiveThink-MoE CLI Interface
==================================

Interactive command-line interface for the MoE system.
This is the user-facing entry point — type prompts, get expert responses.

Features:
- Auto-routing: Prompts are automatically sent to the best expert
- Force expert: Prefix with [code], [math], or [general] to override
- Routing info: Shows which expert was selected and confidence
- Timing: Shows generation time for each response

Usage:
------
    python scripts/cli.py

Commands:
- Type any prompt → auto-routed to best expert
- [code] <prompt> → force code expert
- [math] <prompt> → force math expert
- [general] <prompt> → force general expert
- /experts → show available experts
- /route <text> → show routing decision without generating
- /quit → exit
"""

import os
import sys
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from scripts.pipeline import AdaptiveMoEPipeline


def print_banner():
    """Print the welcome banner."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           AdaptiveThink-MoE Interactive CLI              ║")
    print("║                                                          ║")
    print("║  Mixture-of-Experts with Dynamic LoRA Routing            ║")
    print("║  Experts: Code │ Math │ General                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Commands:")
    print("  Just type a prompt     → Auto-routes to best expert")
    print("  [code] your prompt     → Force code expert")
    print("  [math] your prompt     → Force math expert")
    print("  [general] your prompt  → Force general expert")
    print("  /route <text>          → Show routing only (no generation)")
    print("  /experts               → List available experts")
    print("  /quit or /exit         → Exit")
    print()
    print("─" * 60)


def parse_input(user_input: str) -> tuple:
    """
    Parse user input for force-expert prefix.

    Returns
    -------
    (prompt, force_expert)
        force_expert is None if no prefix, otherwise "code"/"math"/"general"
    """
    # Check for [expert] prefix
    match = re.match(r"\[(code|math|general)\]\s*(.*)", user_input, re.IGNORECASE)
    if match:
        expert = match.group(1).lower()
        prompt = match.group(2)
        return prompt, expert

    return user_input, None


def main():
    """Main CLI loop."""

    print_banner()

    # Initialize the pipeline
    print("Loading pipeline (this may take a moment)...\n")
    try:
        pipeline = AdaptiveMoEPipeline()
    except Exception as e:
        print(f"\nERROR: Failed to initialize pipeline: {e}")
        print("\nMake sure you have:")
        print("  1. Trained all expert adapters (outputs/code_expert/, etc.)")
        print("  2. Trained the router (router/model/)")
        print("  3. All dependencies installed (pip install -r requirements.txt)")
        sys.exit(1)

    print("\nReady! Type your prompt below.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Skip empty input
            if not user_input:
                continue

            # Handle commands
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
                    print(f"\n  Routing: \"{text}\"")
                    print(f"  → Expert: {result['expert'].upper()}")
                    print(f"  → Confidence: {result['confidence']:.1%}")
                    for expert, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                        bar = "█" * int(prob * 20)
                        print(f"    {expert:8s}: {prob:.3f} {bar}")
                    print()
                continue

            # Parse for force-expert prefix
            prompt, force_expert = parse_input(user_input)

            if not prompt:
                continue

            # Generate response
            result = pipeline.generate_with_display(
                prompt,
                force_expert=force_expert,
            )

            print()  # Spacing between interactions

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}\n")
            continue


if __name__ == "__main__":
    main()
