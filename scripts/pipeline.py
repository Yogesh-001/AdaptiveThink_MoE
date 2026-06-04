"""
AdaptiveThink-MoE Inference Pipeline
======================================

This is the CORE of the project — the complete inference system that:
1. Receives a user prompt
2. Routes it to the best expert (via the Router)
3. Loads the appropriate LoRA adapter
4. Generates a response using the specialized expert

This is where the Mixture-of-Experts concept comes together:
- ONE base model (loaded once, ~1GB memory)
- THREE LoRA adapters (~15MB each, loaded/swapped dynamically)
- ONE router (embedding model + classifier, lightweight)

The key innovation compared to traditional MoE:
- Traditional MoE: Multiple full expert networks (expensive)
- Our approach: Shared base + tiny adapters (efficient)
- We get specialization WITHOUT multiplying model size

How adapter swapping works:
----------------------------
PEFT (Parameter-Efficient Fine-Tuning) library supports loading and
unloading LoRA adapters at runtime. The process:
1. Load base model (frozen weights stay in memory)
2. When a request comes in for "code" expert:
   - Load code LoRA adapter (merges small matrices into model)
   - Generate response
3. When next request needs "math" expert:
   - Unload code adapter
   - Load math adapter
   - Generate response

This swap takes <1 second (just loading ~15MB of weights).
Compare to loading a full separate model (~1GB) each time.

Research context:
-----------------
- LoRAHub (Huang et al., 2023): Composes multiple LoRA adapters
- Switch Transformer (Fedus et al., 2022): Token-level routing in MoE
- Our approach: Prompt-level routing + adapter swapping
  (simpler but effective for distinct task domains)
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.router import ExpertRouter


class AdaptiveMoEPipeline:
    """
    The complete Adaptive Mixture-of-Experts inference pipeline.

    This class manages:
    - Base model (loaded once)
    - Expert LoRA adapters (swapped dynamically)
    - Router (classifies prompts → expert selection)
    - Generation (produces responses)

    Usage:
    ------
        pipeline = AdaptiveMoEPipeline()
        result = pipeline.generate("Write a function to sort a list")
        print(result["response"])
        print(result["expert_used"])  # "code"
    """

    # System prompts for each expert (must match training)
    SYSTEM_PROMPTS = {
        "code": "You are a helpful coding assistant. Write clean, correct Python code.",
        "math": (
            "You are a helpful math assistant. Solve problems step by step, "
            "showing your reasoning clearly. End with the final numerical answer."
        ),
        "general": (
            "You are a helpful, knowledgeable assistant. Provide clear, "
            "accurate, and well-organized responses to any question or task."
        ),
    }

    # Default adapter paths
    ADAPTER_PATHS = {
        "code": "outputs/code_expert/adapter",
        "math": "outputs/math_expert/adapter",
        "general": "outputs/general_expert/adapter",
    }

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_paths: dict = None,
        router_model_dir: str = "router/model",
        device: str = None,
    ):
        """
        Initialize the pipeline.

        This loads:
        1. The router (fast, ~80MB embedding model + tiny classifier)
        2. The base LLM (main model, ~1GB)
        3. Expert adapter paths are stored but NOT loaded yet (lazy loading)

        Parameters
        ----------
        base_model_name : str
            HuggingFace model ID for the base model
        adapter_paths : dict, optional
            Override default adapter paths {expert_name: path}
        router_model_dir : str
            Path to the trained router model
        device : str, optional
            "cuda" or "cpu" (auto-detected if None)
        """

        self.base_model_name = base_model_name
        self.adapter_paths = adapter_paths or self.ADAPTER_PATHS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_expert = None  # Track which adapter is currently loaded

        print("=" * 60)
        print("INITIALIZING AdaptiveThink-MoE Pipeline")
        print("=" * 60)
        print(f"  Base model: {base_model_name}")
        print(f"  Device: {self.device}")
        print(f"  Experts: {list(self.adapter_paths.keys())}")

        # Step 1: Load the Router
        print(f"\n[1/3] Loading Router...")
        self.router = ExpertRouter(model_dir=router_model_dir)

        # Step 2: Load Tokenizer
        print(f"\n[2/3] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Load Base Model
        print(f"\n[3/3] Loading Base Model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.base_model = self.base_model.to(self.device)

        # The model starts without any adapter loaded
        self.model = None

        print(f"\n{'=' * 60}")
        print("Pipeline ready!")
        print(f"{'=' * 60}\n")

    def _load_expert(self, expert_name: str):
        """
        Load (or swap) the LoRA adapter for the specified expert.

        This is the key mechanism of our MoE system:
        - If the requested expert is already loaded → do nothing (fast!)
        - If a different expert is loaded → unload it, load new one
        - If no expert is loaded → load from base model

        Why not keep all adapters loaded?
        ---------------------------------
        1. Memory: Each adapter adds parameters to the model
        2. Interference: Multiple active adapters could conflict
        3. PEFT supports only one active adapter at a time (by default)

        In more advanced setups (Phase 7), we could explore:
        - LoRA merging: Weighted combination of multiple adapters
        - Top-K routing: Activate K adapters simultaneously
        """

        if expert_name == self.current_expert:
            # Already loaded — no work needed
            return

        adapter_path = self.adapter_paths.get(expert_name)
        if not adapter_path or not os.path.exists(adapter_path):
            print(f"  WARNING: Adapter not found for '{expert_name}' at '{adapter_path}'")
            print(f"  Falling back to base model (no adapter)")
            self.model = self.base_model
            self.current_expert = None
            return

        # Load the adapter fresh from base model each time
        # (PeftModel doesn't support hot-swapping easily, so we re-wrap)
        print(f"  Loading {expert_name} expert adapter...")
        start_time = time.time()

        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
        )
        self.model.eval()

        load_time = time.time() - start_time
        self.current_expert = expert_name
        print(f"  {expert_name} expert loaded in {load_time:.2f}s")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        force_expert: str = None,
    ) -> dict:
        """
        Generate a response for a given prompt.

        Full pipeline:
        1. Route prompt → determine expert
        2. Load appropriate LoRA adapter
        3. Format prompt with expert-specific system prompt
        4. Generate response
        5. Return response + metadata

        Parameters
        ----------
        prompt : str
            The user's input
        max_new_tokens : int
            Maximum tokens to generate
        temperature : float, optional
            Override temperature (default: expert-specific)
        force_expert : str, optional
            Skip routing and force a specific expert ("code"/"math"/"general")

        Returns
        -------
        dict with:
            - response: str (generated text)
            - expert_used: str (which expert handled this)
            - routing_info: dict (router probabilities and confidence)
            - generation_time: float (seconds)
        """

        total_start = time.time()

        # Step 1: ROUTE the prompt
        if force_expert:
            routing_info = {
                "expert": force_expert,
                "confidence": 1.0,
                "probabilities": {force_expert: 1.0},
                "threshold_applied": False,
            }
            expert_name = force_expert
        else:
            routing_info = self.router.route(prompt)
            expert_name = routing_info["expert"]

        # Step 2: LOAD the expert adapter
        self._load_expert(expert_name)

        # Step 3: FORMAT the prompt with expert-specific system prompt
        system_prompt = self.SYSTEM_PROMPTS.get(expert_name, self.SYSTEM_PROMPTS["general"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Step 4: GENERATE response
        # Temperature defaults vary by expert:
        # - Code: 0.2 (deterministic, correctness matters)
        # - Math: 0.1 (very deterministic, precision matters)
        # - General: 0.5 (some creativity allowed)
        if temperature is None:
            temperature = {"code": 0.2, "math": 0.1, "general": 0.5}.get(expert_name, 0.5)

        gen_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        # Decode only new tokens (not the input)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_time = time.time() - gen_start
        total_time = time.time() - total_start

        return {
            "response": response,
            "expert_used": expert_name,
            "routing_info": routing_info,
            "generation_time": generation_time,
            "total_time": total_time,
        }

    def generate_with_display(self, prompt: str, **kwargs) -> dict:
        """
        Generate and display the result in a formatted way.
        Convenience method for CLI/demo use.
        """

        result = self.generate(prompt, **kwargs)

        # Display routing info
        routing = result["routing_info"]
        print(f"\n  ┌─ Router Decision ─────────────────────────────")
        print(f"  │ Expert: {result['expert_used'].upper()}")
        print(f"  │ Confidence: {routing['confidence']:.1%}")
        if "probabilities" in routing:
            for expert, prob in sorted(routing["probabilities"].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"  │   {expert:8s}: {prob:.3f} {bar}")
        print(f"  └────────────────────────────────────────────────")

        # Display response
        print(f"\n  Response ({result['generation_time']:.1f}s):")
        print(f"  {'─' * 50}")
        # Indent response for readability
        for line in result["response"].split("\n"):
            print(f"  {line}")
        print(f"  {'─' * 50}")
        print(f"  Total time: {result['total_time']:.1f}s")

        return result
