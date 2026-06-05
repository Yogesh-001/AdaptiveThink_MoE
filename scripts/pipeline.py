"""AdaptiveThink-MoE Inference Pipeline — routes prompts and generates with specialized experts."""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.router import ExpertRouter


class AdaptiveMoEPipeline:
    """Complete MoE inference pipeline: router + dynamic LoRA adapter swapping."""

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
        self.base_model_name = base_model_name
        self.adapter_paths = adapter_paths or self.ADAPTER_PATHS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_expert = None

        print("=" * 60)
        print("INITIALIZING AdaptiveThink-MoE Pipeline")
        print("=" * 60)
        print(f"  Base model: {base_model_name}")
        print(f"  Device: {self.device}")

        # Load router
        print(f"
[1/3] Loading Router...")
        self.router = ExpertRouter(model_dir=router_model_dir)

        # Load tokenizer
        print(f"
[2/3] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"
[3/3] Loading Base Model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.base_model = self.base_model.to(self.device)

        self.model = None
        print(f"
{"=" * 60}")
        print(f"{"=" * 60}
")

    def _load_expert(self, expert_name: str):
        """Load or swap the LoRA adapter for the specified expert."""
        if expert_name == self.current_expert:
            return

        adapter_path = self.adapter_paths.get(expert_name)
        if not adapter_path or not os.path.exists(adapter_path):
            print(f"  WARNING: Adapter not found for '{expert_name}', using base model")
            self.model = self.base_model
            self.current_expert = None
            return

        print(f"  Loading {expert_name} expert adapter...")
        start_time = time.time()
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()
        self.current_expert = expert_name
        print(f"  Loaded in {time.time() - start_time:.2f}s")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        force_expert: str = None,
    ) -> dict:
        """Generate a response: route prompt -> load adapter -> generate."""
        total_start = time.time()

        # Route
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

        # Load adapter
        self._load_expert(expert_name)

        # Format prompt
        system_prompt = self.SYSTEM_PROMPTS.get(expert_name, self.SYSTEM_PROMPTS["general"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate
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

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "response": response,
            "expert_used": expert_name,
            "routing_info": routing_info,
            "generation_time": time.time() - gen_start,
            "total_time": time.time() - total_start,
        }

    def generate_with_display(self, prompt: str, **kwargs) -> dict:
        """Generate and display result with routing info."""
        result = self.generate(prompt, **kwargs)
        routing = result["routing_info"]

        print(f"
  ┌─ Router Decision ─────────────────────────────")
        print(f"  │ Expert: {result["expert_used"].upper()}")
        print(f"  │ Confidence: {routing["confidence"]:.1%}")
        if "probabilities" in routing:
            for expert, prob in sorted(routing["probabilities"].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"  │   {expert:8s}: {prob:.3f} {bar}")
        print(f"  └────────────────────────────────────────────────")

        print(f"
  Response ({result["generation_time"]:.1f}s):")
        print(f"  {"─" * 50}")
        for line in result["response"].split("
"):
            print(f"  {line}")
        print(f"  {"─" * 50}")
        print(f"  Total time: {result["total_time"]:.1f}s")

        return result
