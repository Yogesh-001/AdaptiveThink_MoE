"""
Router Module
==============

This is the inference-time router. Given a user prompt, it:
1. Encodes the prompt into an embedding
2. Classifies it into code/math/general
3. Returns the predicted expert + confidence score

This module is imported by the inference pipeline (Phase 5)
to decide which LoRA adapter to load for each query.

Usage:
------
    from router.router import ExpertRouter

    router = ExpertRouter()
    result = router.route("Write a function to sort a list")
    # result = {"expert": "code", "confidence": 0.95, "probabilities": {...}}
"""

import os
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer


class ExpertRouter:
    """
    Routes user prompts to the appropriate expert.

    The router uses:
    1. A sentence embedding model (all-MiniLM-L6-v2) for semantic understanding
    2. A trained classifier (LogisticRegression) for expert prediction

    Design decisions:
    -----------------
    - Stateless: Each call is independent (no conversation history needed)
    - Fast: ~10ms per routing decision (embedding + classify)
    - Confidence-aware: Returns probability scores for all experts
    - Threshold-based: Can fall back to "general" if confidence is low

    The confidence threshold is important:
    - If the router is >70% confident → use predicted expert
    - If confidence is low → might be ambiguous, default to general
    - This prevents bad routing on edge cases
    """

    def __init__(
        self,
        model_dir: str = "router/model",
        embedding_model_name: str = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the router.

        Parameters
        ----------
        model_dir : str
            Directory containing the trained classifier and label encoder
        embedding_model_name : str, optional
            Override the embedding model (default: read from config)
        confidence_threshold : float
            Minimum confidence to trust the prediction (else default to general)
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold

        # Load router config
        config_path = os.path.join(model_dir, "router_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load the embedding model
        model_name = embedding_model_name or self.config["embedding_model"]
        print(f"Loading router embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Load the trained classifier
        classifier_path = os.path.join(model_dir, "router_classifier.joblib")
        self.classifier = joblib.load(classifier_path)

        # Load the label encoder (maps indices back to expert names)
        encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        self.label_encoder = joblib.load(encoder_path)

        print(f"Router loaded. Experts: {list(self.label_encoder.classes_)}")
        print(f"Confidence threshold: {self.confidence_threshold}")

    def route(self, prompt: str) -> dict:
        """
        Route a single prompt to the best expert.

        Parameters
        ----------
        prompt : str
            The user's input prompt

        Returns
        -------
        dict with keys:
            - expert: str ("code", "math", or "general")
            - confidence: float (0-1, probability of predicted class)
            - probabilities: dict mapping expert names to their probabilities
            - threshold_applied: bool (whether threshold forced a default)
        """

        # Step 1: Encode prompt into embedding vector
        # Shape: (1, 384) — one sample, 384 dimensions
        embedding = self.encoder.encode([prompt])

        # Step 2: Get class probabilities from classifier
        # predict_proba returns shape (1, n_classes)
        probabilities = self.classifier.predict_proba(embedding)[0]

        # Step 3: Find the predicted class (highest probability)
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probabilities[predicted_idx]

        # Step 4: Apply confidence threshold
        # If confidence is below threshold, default to "general"
        # This handles ambiguous prompts that might confuse the router
        threshold_applied = False
        if confidence < self.confidence_threshold:
            predicted_label = "general"
            threshold_applied = True

        # Build probability map: {expert_name: probability}
        prob_map = {
            label: float(prob)
            for label, prob in zip(self.label_encoder.classes_, probabilities)
        }

        return {
            "expert": predicted_label,
            "confidence": float(confidence),
            "probabilities": prob_map,
            "threshold_applied": threshold_applied,
        }

    def route_batch(self, prompts: list) -> list:
        """
        Route multiple prompts at once (more efficient than one-by-one).

        Batched encoding is faster because the embedding model can
        process multiple texts in parallel on GPU.
        """
        embeddings = self.encoder.encode(prompts)
        probabilities = self.classifier.predict_proba(embeddings)

        results = []
        for i, (prompt, probs) in enumerate(zip(prompts, probabilities)):
            predicted_idx = np.argmax(probs)
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence = probs[predicted_idx]

            threshold_applied = False
            if confidence < self.confidence_threshold:
                predicted_label = "general"
                threshold_applied = True

            prob_map = {
                label: float(prob)
                for label, prob in zip(self.label_encoder.classes_, probs)
            }

            results.append({
                "expert": predicted_label,
                "confidence": float(confidence),
                "probabilities": prob_map,
                "threshold_applied": threshold_applied,
            })

        return results


def demo():
    """Interactive demo of the router."""

    router = ExpertRouter()

    print("\n" + "=" * 60)
    print("ROUTER DEMO — Type a prompt to see which expert handles it")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        print()
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        result = router.route(prompt)

        print(f"\n  Expert: {result['expert'].upper()}")
        print(f"  Confidence: {result['confidence']:.1%}")
        if result["threshold_applied"]:
            print(f"  (Low confidence — defaulted to general)")
        print(f"  Probabilities:")
        for expert, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {expert:8s}: {prob:.3f} {bar}")


if __name__ == "__main__":
    demo()
